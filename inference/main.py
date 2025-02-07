import re
import os
import json
import logging
import time
import sys
import pandas as pd
from bs4 import BeautifulSoup

# -------------- Add these two lines for caching --------------
import langchain
from langchain_community.cache import InMemoryCache
# -------------------------------------------------------------

import config
from metric import BenchmarkEvaluator
from model_loader import BaseModel

logging.basicConfig(level=logging.INFO)

# -------------- Initialize the LangChain LLM cache --------------
langchain.llm_cache = InMemoryCache()
# ---------------------------------------------------------------

def clean_html(raw_html: str) -> str:
    """
    Minimal HTML-to-text cleaning with BeautifulSoup.
    Removes <script> and <style>, then collapses whitespace.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text

def chunk_text(text: str, chunk_size: int):
    """
    Splits `text` into a list of substrings, each at most `chunk_size` characters.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def load_document_text(doc_id: str) -> list[str]:
    """
    Load HTML from the JSON file, clean it.
    If longer than config.MAX_CHAR_FOR_SYSTEM, chunk it; else return as single chunk.
    Returns a list of chunks (one or more).
    """
    json_path = os.path.join(config.JSON_PATH, config.JSON_FILE)
    if not os.path.exists(json_path):
        logging.error(f"JSON file not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                if str(entry.get("id")) == str(doc_id):
                    raw_html = entry.get("data", {}).get("html", "")
                    cleaned = clean_html(raw_html)
                    if not cleaned:
                        logging.warning(f"Doc {doc_id} is empty after cleaning.")
                        return []

                    if len(cleaned) > config.MAX_CHAR_FOR_SYSTEM:
                        logging.warning(
                            f"Doc {doc_id} length {len(cleaned)} > {config.MAX_CHAR_FOR_SYSTEM}, chunking..."
                        )
                        chunks = chunk_text(cleaned, config.MAX_CHAR_FOR_SYSTEM)
                        logging.info(
                            f"Doc {doc_id} chunked into {len(chunks)} parts. Each up to {config.MAX_CHAR_FOR_SYSTEM} chars."
                        )
                        return chunks
                    else:
                        logging.info(
                            f"Doc {doc_id} loaded, length={len(cleaned)} chars."
                        )
                        return [cleaned]  # return as a single-element list
            logging.warning(
                f"Document {doc_id} not found in {json_path}. Returning empty list."
            )
            return []
    except Exception as e:
        logging.error(f"Error reading {json_path}: {e}")
        return []

def build_prompt_single(document_text: str, question: str, question_index: int) -> list[dict]:
    """
    Single user message combining instructions + doc text + question.
    Enforce returning only JSON.
    """
    user_instructions = (
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to answer "
        "the question given to you about the provided credit agreement. "
        "If you believe the answer is not present in the agreement, say 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Apple Inc. is a technology company headquartered in Cupertino, California. "
        "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.\n\n"

        "The given question:\n"
        "Q1: Where is the headquarters of Apple Inc.?\n\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "Cupertino, California"}\n'
        "  ]\n"
        "}\n\n"
        
        "[USER INPUT]\n"
        f"Document:\n{document_text}\n\n"

        "[QUESTION]\n"
        f"Q{question_index}: {question}\n"
    )

    return [{"role": "user", "content": user_instructions}]


def build_prompt_batch(document_text: str, questions: list[str]) -> list[dict]:
    """
    One user message that includes multiple questions at once.
    Enforce returning only JSON.
    """
    prompt_lines = [
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to answer "
        "the questions given to you in batches about the provided credit agreement. "
        "If you believe the answer is not present in the agreement, say 'Not found'.\n\n"
        
        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Tesla, Inc. is an American electric vehicle and clean energy company founded in 2003 by Martin Eberhard and Marc Tarpenning. "
        "Elon Musk became the largest investor and later CEO.\n\n"

        "The given questions:\n"
        "Q1: Who founded Tesla?\n"
        "Q2: What year was Tesla founded?\n\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "Martin Eberhard, Marc Tarpenning"},\n'
        '    {"question_index": 2, "answer": "2003"}\n'
        "  ]\n"
        "}\n\n"
        
        "[USER INPUT]\n"
        f"Document:\n{document_text}\n\n"

        "[QUESTIONS]\n"
    ]

    for i, question in enumerate(questions, start=1):
        prompt_lines.append(f"Q{i}: {question}")

    combined_prompt = "\n".join(prompt_lines)
    return [{"role": "user", "content": combined_prompt}]



def build_prompt_combine_answers(partial_answers: list[str], questions: list[str]) -> list[dict]:
    """
    Build a prompt to merge/combine partial answers (in JSON form) from multiple chunks into final answers.
    We'll ask the LLM to output in the same JSON format:
    {
      "answers": [
        {"question_index": 1, "answer": "..."},
        ...
      ]
    }
    Enforce returning only JSON.
    """
    prompt_lines = [
        "[SYSTEM INPUT]\n"
        "You are a financial expert, and your task is to combine or merge "
        "the provided partial answers, coming from different chunks of a credit agreement, "
        "into a single final answer for each of the questions given to you. "
        "If you believe the answer is not present in the agreement, say 'Not found'.\n\n"
        
        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "The given document:\n"
        "Tesla, Inc. is an American electric vehicle and clean energy company founded in 2003 by Martin Eberhard and Marc Tarpenning. "
        "Elon Musk became the largest investor and later CEO.\n\n"

        "The given questions:\n"
        "Q1: Who founded Tesla?\n"
        "Q2: What year was Tesla founded?\n\n"

        "The expected output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "Martin Eberhard, Marc Tarpenning"},\n'
        '    {"question_index": 2, "answer": "2003"}\n'
        "  ]\n"
        "}\n\n"

        "[USER INPUT]\n"
    ]

    for i, ans in enumerate(partial_answers, start=1):
        prompt_lines.append(f"Chunk {i} partial answer JSON:\n{ans}\n")

    prompt_lines.append("\n[QUESTIONS]\n")
    for i, q in enumerate(questions, start=1):
        prompt_lines.append(f"Q{i}: {q}")

    combined_prompt = "\n".join(prompt_lines)
    return [{"role": "user", "content": combined_prompt}]

def parse_llm_json(raw_response: str, num_questions: int) -> dict:
    """
    Expects a JSON string, possibly embedded in extra text, like:
    
    Some preamble text...
    ```json
    {
      "answers": [
        {"question_index": 1, "answer": "..."},
        ...
      ]
    }
    ```
    Some additional commentary...
    
    Returns a dict: {1: "answer1", 2: "answer2", ...}
    If invalid JSON or missing fields, defaults to "LLM parse error".
    """
    default_result = {i: "LLM parse error" for i in range(1, num_questions + 1)}
    
    # Extract JSON from inside triple backticks
    match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL)
    if match:
        cleaned_response = match.group(1).strip()
    else:
        cleaned_response = raw_response.strip()
    
    logging.debug(f"Parsing LLM JSON: {cleaned_response}")  
    
    try:
        data = json.loads(cleaned_response)
        if "answers" not in data:
            logging.warning("No 'answers' key found in the JSON response.")
            return default_result
        
        answers = data["answers"]
        for ans in answers:
            idx = ans.get("question_index")
            content = ans.get("answer", "")
            if isinstance(idx, int) and 1 <= idx <= num_questions:
                default_result[idx] = content
        
        return default_result
    except json.JSONDecodeError as e:
        logging.warning(f"JSON parse error: {e}")
        return default_result


def call_llm_with_retries(llm, messages: list[dict], extra_log_info: str = "") -> str:
    """
    Call the LLM up to config.NUM_RETRIES times if blank is returned.
    Then we do a second layer of retries if we fail JSON parsing.
    
    We return the *raw string* from LLM (which should be JSON).
    extra_log_info can be used to log chunk/question context, etc.
    """
    # First, up to config.NUM_RETRIES attempts for non-empty response
    for attempt in range(config.NUM_RETRIES):
        try:
            prompt_str = messages[0]['content'] if messages else ""
            logging.info(
                f"LLM call attempt {attempt+1}/{config.NUM_RETRIES} {extra_log_info} "
                f"(prompt length: {len(prompt_str)} chars)"
            )

            response = llm.invoke(messages)
            raw_output = response.content.strip() if hasattr(response, "content") else str(response).strip()
            print(raw_output)

            if raw_output:
                return raw_output
            else:
                logging.warning(f"Got an empty response from LLM. Retrying in 1s...")
                time.sleep(1.0)
        except Exception as e:
            logging.error(f"LLM call error on attempt {attempt+1}: {e}")
            time.sleep(1.0)

    # If all attempts yield empty string:
    logging.error("All attempts returned empty response. Giving up.")
    return ""

def get_llm_json_response(llm, messages: list[dict], num_questions: int, extra_log_info: str) -> dict:
    """
    Attempts to get a valid JSON parse from the LLM.
    Retries multiple times (config.NUM_RETRIES) if the JSON parse fails.
    """
    parsed_result = {}
    for parse_attempt in range(config.NUM_RETRIES):
        raw_output = call_llm_with_retries(llm, messages, extra_log_info=extra_log_info)
        if not raw_output:
            # If we got no output, skip parse & just retry
            logging.warning(f"Empty output from LLM (parse attempt {parse_attempt+1}). Retrying...")
            time.sleep(1.0)
            continue

        parsed_result = parse_llm_json(raw_output, num_questions)
        # Check if parse was successful
        if any(ans != "LLM parse error" for ans in parsed_result.values()):
            return parsed_result
        else:
            logging.warning(f"Parse error (parse attempt {parse_attempt+1}). Retrying LLM call...")

    # If all parse attempts fail, return the parse_result with "LLM parse error"
    return parsed_result

def main():
    # 1) Load the model
    model_loader = BaseModel(
        llm_provider=config.LLM_PROVIDER,
        model_name=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        max_tokens=config.max_tokens_generation
    )
    model_loader.load()
    llm = model_loader.get_model()

    # 2) Read the CSV
    input_csv_path = os.path.join(config.INPUT_PATH, f"{config.QUESTION_FILE}.csv")
    if not os.path.exists(input_csv_path):
        logging.error(f"Input CSV not found: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)

    # Ensure we have an "llm_response" column to store results
    if "llm_response" not in df.columns:
        df["llm_response"] = ""

    # Group by document_number
    grouped = df.groupby("document_number")

    # 3) For each document group, load text & ask the LLM
    for doc_id, group_indices in grouped.groups.items():
        indices_list = list(group_indices)
        doc_chunks = load_document_text(str(doc_id))  # list of text chunks
        if not doc_chunks:
            logging.warning(f"Document {doc_id} is empty. Setting llm_response='No doc text'.")
            for idx in indices_list:
                df.at[idx, "llm_response"] = "No doc text"
            continue

        questions = df.loc[indices_list, "question"].tolist()
        num_questions = len(questions)
        logging.info(f"Processing doc_id={doc_id} with {num_questions} questions...")

        # If there's only 1 chunk, proceed as before (no chunk merging needed).
        if len(doc_chunks) == 1:
            single_chunk_text = doc_chunks[0]

            # If context_chat is True => each question is a separate prompt
            if config.context_chat:
                for i, row_idx in enumerate(indices_list, start=1):
                    question_text = df.at[row_idx, "question"]
                    log_msg = f"[doc={doc_id} chunk=1 question_index={i}]"
                    logging.info(f"Q{i}/{num_questions} => {question_text}")

                    # Build prompt
                    messages = build_prompt_single(single_chunk_text, question_text, i)
                    # Attempt to get valid JSON
                    parsed_answers = get_llm_json_response(llm, messages, 1, extra_log_info=log_msg)
                    df.at[row_idx, "llm_response"] = parsed_answers[1]
            else:
                # Single prompt for all questions at once
                log_msg = f"[doc={doc_id} chunk=1 batch_mode]"
                messages = build_prompt_batch(single_chunk_text, questions)
                parsed_answers = get_llm_json_response(llm, messages, num_questions, extra_log_info=log_msg)
                for i, row_idx in enumerate(indices_list, start=1):
                    df.at[row_idx, "llm_response"] = parsed_answers[i]

        else:
            # Multiple chunks => gather partial answers from each chunk, then combine
            if config.context_chat:
                # Each question is separate across all chunks
                for i, row_idx in enumerate(indices_list, start=1):
                    question_text = df.at[row_idx, "question"]
                    logging.info(f"Q{i}/{num_questions} => {question_text}")
                    partial_responses = []

                    for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                        log_msg = f"[doc={doc_id} chunk={c_idx} question_index={i}]"
                        messages_chunk = build_prompt_single(chunk_text, question_text, i)
                        # get partial JSON
                        chunk_parsed_answers = get_llm_json_response(llm, messages_chunk, 1, extra_log_info=log_msg)

                        # Convert it back to string (so we can combine later). We'll store raw JSON string:
                        # We'll just dump the chunk_parsed_answers to JSON string for the combine stage
                        # but if it's "LLM parse error", let's store a placeholder
                        if "LLM parse error" in chunk_parsed_answers[1]:
                            partial_responses.append('{"answers":[{"question_index":1,"answer":"LLM parse error"}]}')
                        else:
                            partial_json_str = json.dumps({
                                "answers": [
                                    {"question_index": 1, "answer": chunk_parsed_answers[1]}
                                ]
                            })
                            partial_responses.append(partial_json_str)

                    # Now combine partial responses for question i
                    combine_msg = f"[doc={doc_id} combine question_index={i}]"
                    combine_prompt = build_prompt_combine_answers(partial_responses, [question_text])
                    combined_final = get_llm_json_response(llm, combine_prompt, 1, extra_log_info=combine_msg)
                    df.at[row_idx, "llm_response"] = combined_final[1]

            else:
                # Batch mode: all questions at once across each chunk
                partial_responses = []
                for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                    log_msg = f"[doc={doc_id} chunk={c_idx} batch_mode]"
                    messages_chunk = build_prompt_batch(chunk_text, questions)
                    chunk_parsed_answers = get_llm_json_response(llm, messages_chunk, num_questions, extra_log_info=log_msg)

                    # Convert chunk_parsed_answers to a JSON string
                    # If parse error, keep placeholders for each question
                    if any(ans == "LLM parse error" for ans in chunk_parsed_answers.values()):
                        fake_json = {
                            "answers": [
                                {"question_index": i, "answer": "LLM parse error"}
                                for i in range(1, num_questions + 1)
                            ]
                        }
                        partial_responses.append(json.dumps(fake_json))
                    else:
                        # Build the minimal JSON we want to pass to combine
                        partial_json = {"answers": []}
                        for q_idx in range(1, num_questions + 1):
                            partial_json["answers"].append(
                                {"question_index": q_idx, "answer": chunk_parsed_answers[q_idx]}
                            )
                        partial_responses.append(json.dumps(partial_json))

                # Combine partial JSON answers
                combine_msg = f"[doc={doc_id} combine batch_mode]"
                combine_prompt = build_prompt_combine_answers(partial_responses, questions)
                combined_output = get_llm_json_response(llm, combine_prompt, num_questions, extra_log_info=combine_msg)
                for i, row_idx in enumerate(indices_list, start=1):
                    df.at[row_idx, "llm_response"] = combined_output[i]

    # 4) Save results
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize the model name for file naming
    sanitized_model_name = config.MODEL_NAME.replace("/", "-")
    # Remove other invalid filename chars
    sanitized_model_name = re.sub(r'[<>:"/\\|?*]', '-', sanitized_model_name)

    output_csv = f"{config.QUESTION_FILE}_{sanitized_model_name}.csv"
    output_path = os.path.join(output_dir, output_csv)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved LLM answers to {output_path}")

    # 5) Evaluate metrics
    evaluator = BenchmarkEvaluator(results_dir=config.OUTPUT_PATH, metrics_dir=config.METRICS_PATH)
    evaluator.evaluate_all()
    logging.info(f"Saved metrics to {evaluator.metrics_dir}")


if __name__ == "__main__":
    main()
