# main.py

import re
import os
import json
import logging
import time
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
    """
    user_instructions = (
        "You are a world-class AI system and a helpful assistant.\n"
        "You have the following document text.\n"
        "If the answer is not found, say 'Not found'.\n\n"
        f"Document:\n{document_text}\n\n"
        "Please return your answer in JSON format EXACTLY:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "..."}\n'
        "  ]\n"
        "}\n\n"
        f"Question (index={question_index}): {question}"
    )
    return [{"role": "user", "content": user_instructions}]

def build_prompt_batch(document_text: str, questions: list[str]) -> list[dict]:
    """
    One user message that includes multiple questions at once.
    """
    prompt_lines = [
        "You are a world-class AI system and a helpful assistant.",
        "You have the following document text.",
        "If the answer is not found, say 'Not found'.\n",
        f"Document:\n{document_text}\n",
        "Return your answers in JSON format EXACTLY.\n"
        "Do not return more text than necessary, just the answers.\n",
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "..."},\n'
        '    {"question_index": 2, "answer": "..."},\n'
        "    ...\n"
        "  ]\n"
        "}\n",
        "Here are the questions:"
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
    """
    # We'll present the partial answers to the LLM as strings. 
    # We keep the final result in the same JSON structure for easier parsing.
    prompt_lines = [
        "You are a world-class AI system and a helpful assistant.",
        "We have multiple partial answers for the same questions, coming from different chunks of the document.\n"
        "Your task is to combine or merge these partial answers into a single final answer for each question.\n"
        "If the answer is still not found, say 'Not found'.\n",
        "Partial answers (in JSON) from each chunk:\n"
    ]

    for i, ans in enumerate(partial_answers, start=1):
        prompt_lines.append(f"Chunk {i} partial answer JSON:\n{ans}\n")

    prompt_lines.append(
        "Now combine/merge them carefully and return the final answers in the SAME JSON format EXACTLY:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "..."},\n'
        '    {"question_index": 2, "answer": "..."},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
    )

    # Also restate the questions for clarity
    prompt_lines.append("Here are the original questions:")
    for i, q in enumerate(questions, start=1):
        prompt_lines.append(f"Q{i}: {q}")

    combined_prompt = "\n".join(prompt_lines)
    return [{"role": "user", "content": combined_prompt}]

def parse_llm_json(raw_response: str, num_questions: int) -> dict:
    """
    Expects a JSON string like:
    {
      "answers": [
        {"question_index": 1, "answer": "..."},
        ...
      ]
    }
    Returns a dict: {1: "answer1", 2: "answer2", ...}
    If invalid JSON or missing fields, default to "LLM parse error".
    """
    default_result = {i: "LLM parse error" for i in range(1, num_questions + 1)}
    try:
        # Remove any stray triple-backtick fences
        cleaned_response = re.sub(r"```json\s*|\s*```", "", raw_response).strip()
        print(cleaned_response)
        data = json.loads(cleaned_response)
        if "answers" not in data:
            return default_result
        answers = data["answers"]
        for ans in answers:
            idx = ans.get("question_index")
            content = ans.get("answer", "")
            if isinstance(idx, int) and 1 <= idx <= num_questions:
                default_result[idx] = content
        return default_result
    except json.JSONDecodeError:
        return default_result

def call_llm_with_retries(llm, messages: list[dict]) -> str:
    """
    Call the LLM up to config.NUM_RETRIES times if blank or invalid JSON is returned.
    We return the *raw string* from LLM (which should be JSON).
    """
    for attempt in range(config.NUM_RETRIES):
        try:
            response = llm.invoke(messages)
            # Grab the text from `response.content` if using ChatOpenAI or a similar interface
            raw_output = response.content.strip() if hasattr(response, "content") else str(response).strip()
            if raw_output:
                return raw_output
            else:
                logging.warning(f"Got an empty response from LLM. Attempt {attempt+1}/{config.NUM_RETRIES}. Retrying...")
                time.sleep(1.0)

        except Exception as e:
            logging.error(f"LLM call error (attempt {attempt+1}): {e}")
            time.sleep(1.0)

    # If all attempts fail or yield empty:
    return ""

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
            # No doc text
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
                    logging.info(f"Q{i}/{num_questions} => {question_text}")

                    messages = build_prompt_single(single_chunk_text, question_text, i)
                    raw_output = call_llm_with_retries(llm, messages)
                    if not raw_output:
                        df.at[row_idx, "llm_response"] = "LLM error or empty"
                        continue

                    parsed_answers = parse_llm_json(raw_output, 1)
                    df.at[row_idx, "llm_response"] = parsed_answers[1]

            else:
                # Single prompt for all questions at once
                messages = build_prompt_batch(single_chunk_text, questions)
                raw_output = call_llm_with_retries(llm, messages)
                if not raw_output:
                    for i, row_idx in enumerate(indices_list, start=1):
                        df.at[row_idx, "llm_response"] = "LLM error or empty"
                    continue

                parsed_answers = parse_llm_json(raw_output, num_questions)
                for i, row_idx in enumerate(indices_list, start=1):
                    df.at[row_idx, "llm_response"] = parsed_answers[i]

        else:
            # Multiple chunks. We'll gather partial answers for each chunk, 
            # then do a final "combine" call to produce the final answer.

            # -- If context_chat: each question is separate across all chunks --
            if config.context_chat:
                # We'll store final answers in the dataframe, 
                # but first we gather partial answers from each chunk for each question
                for i, row_idx in enumerate(indices_list, start=1):
                    question_text = df.at[row_idx, "question"]
                    logging.info(f"Q{i}/{num_questions} => {question_text}")

                    partial_responses = []
                    # Get partial answer from each chunk
                    for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                        messages_chunk = build_prompt_single(chunk_text, question_text, i)
                        raw_output_chunk = call_llm_with_retries(llm, messages_chunk)

                        if not raw_output_chunk:
                            # If the chunk returned nothing, keep a placeholder
                            partial_responses.append('{"answers":[{"question_index":1,"answer":"LLM error"}]}')
                        else:
                            partial_responses.append(raw_output_chunk)

                    # Now we combine partial responses for question i
                    combine_prompt = build_prompt_combine_answers(partial_responses, [question_text])
                    combined_output = call_llm_with_retries(llm, combine_prompt)

                    if not combined_output:
                        df.at[row_idx, "llm_response"] = "LLM combine error or empty"
                        continue

                    # Parse the final merged answer for that single question
                    parsed_final = parse_llm_json(combined_output, 1)
                    df.at[row_idx, "llm_response"] = parsed_final[1]

            else:
                # -- Batch mode: we handle all questions at once across each chunk --
                partial_responses = []
                for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                    messages_chunk = build_prompt_batch(chunk_text, questions)
                    raw_output_chunk = call_llm_with_retries(llm, messages_chunk)
                    if not raw_output_chunk:
                        # If the chunk returned nothing, keep a placeholder 
                        # with the correct structure for num_questions
                        fake_json = {
                            "answers": [
                                {"question_index": i, "answer": "LLM error"} 
                                for i in range(1, num_questions+1)
                            ]
                        }
                        partial_responses.append(json.dumps(fake_json))
                    else:
                        partial_responses.append(raw_output_chunk)

                # Now we have partial JSON answers from all chunks
                combine_prompt = build_prompt_combine_answers(partial_responses, questions)
                combined_output = call_llm_with_retries(llm, combine_prompt)
                if not combined_output:
                    # If final merge fails, set error
                    for i, row_idx in enumerate(indices_list, start=1):
                        df.at[row_idx, "llm_response"] = "LLM combine error or empty"
                    continue

                # Parse the final merged answers
                parsed_final = parse_llm_json(combined_output, num_questions)
                for i, row_idx in enumerate(indices_list, start=1):
                    df.at[row_idx, "llm_response"] = parsed_final[i]

    # 4) Save results
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_csv = f"{config.QUESTION_FILE}_{config.MODEL_NAME}.csv"
    output_path = os.path.join(output_dir, output_csv)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved LLM answers to {output_path}")

    # 5) Evaluate metrics
    evaluator = BenchmarkEvaluator(results_dir=config.OUTPUT_PATH, metrics_dir=config.METRICS_PATH)
    evaluator.evaluate_all()
    logging.info(f"Saved metrics to {evaluator.metrics_dir}")


if __name__ == "__main__":
    main()
