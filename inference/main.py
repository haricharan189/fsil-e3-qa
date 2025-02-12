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
        "You are an expert in financial documents. Your task is to answer multiple questions in one batch, "
        "based solely on the provided credit agreement text.\n\n"

        "Answering Rules:\n"
        "1. If the answer is explicitly found in the document, extract it exactly as written.\n"
        "2. If the answer is not found in the document, respond with: 'Not found'.\n"
        "3. Do not provide any extra explanation, reasoning, or assumptions.\n"

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
    Constructs a single user message that includes multiple questions about a credit agreement.
    The response must be in valid JSON format only.
    """
    prompt_lines = [
        "[SYSTEM INPUT]\n"
        "You are an expert in financial documents. Your task is to answer multiple questions in one batch, "
        "based solely on the provided credit agreement text.\n\n"

        "Answering Rules:\n"
        "1. If the answer is explicitly found in the document, extract it exactly as written.\n"
        "2. If the answer is not found in the document, respond with: 'Not found'.\n"
        "3. Do not provide any extra explanation, reasoning, or assumptions.\n"
        
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
        "You are an expert in financial documents. Your task is to merge multiple partial answers "
        "from different chunks of a credit agreement into a single, final answer for each question.\n\n"

        "Rules for merging answers:\n"
        "1. If a chunk provides an answer and another says 'Not found', use the provided answer.\n"
        "2. If multiple chunks provide different answers, merge them into a single, coherent response.\n"
        "3. If multiple chunks provide the same answer, retain it as-is.\n"
        "4. If all chunks say 'Not found', the final answer should be 'Not found'.\n\n"

        "[EXPECTED OUTPUT]\n"
        "Respond ONLY with valid JSON, nothing else. See the example below.\n\n"

        "Example input (chunks with partial answers):\n"
        "Chunk 1 partial answer JSON:\n"
        '{ "answers": [{"question_index": 1, "answer": "UBS AG, STAMFORD BRANCH"},'
        '{"question_index": 2, "answer": "KeyBank National Association"},'
        '{"question_index": 3, "answer": "Not found"}] }\n\n'

        "Chunk 2 partial answer JSON:\n"
        '{ "answers": [{"question_index": 1, "answer": "Not found"}, '
        '{"question_index": 2, "answer": "KeyBank National Association"},'
        '{"question_index": 3, "answer": "Not found"}] }\n\n'

        "Expected merged output:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question_index": 1, "answer": "UBS AG, STAMFORD BRANCH"},\n'
        '    {"question_index": 2, "answer": "KeyBank National Association"},\n'
        '    {"question_index": 3, "answer": "Not found"}\n'
        "  ]\n"
        "}\n\n"

        "[USER INPUT]\n"
        "Below are the partial answers from different chunks:\n"
    ]

    for i, ans in enumerate(partial_answers, start=1):
        prompt_lines.append(f"Chunk {i} partial answer JSON:\n{ans}\n")

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
    We return the *raw string* from LLM (which should be JSON).
    extra_log_info can be used to log chunk/question context, etc.
    """
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
                if config.WAIT_TIME_ENABLED:
                    time.sleep(config.WAIT_TIME_DURATION)
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
            logging.warning(f"Empty output from LLM (parse attempt {parse_attempt+1}). Retrying...")
            time.sleep(1.0)
            continue

        parsed_result = parse_llm_json(raw_output, num_questions)
        # Check if parse was successful
        if any(ans != "LLM parse error" for ans in parsed_result.values()):
            return parsed_result
        else:
            logging.warning(f"Parse error (parse attempt {parse_attempt+1}). Retrying LLM call...")

    return parsed_result  # returns final parse_result with "LLM parse error" if not fixed


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
    all_doc_ids = list(grouped.groups.keys())

    total_docs = len(all_doc_ids)
    processed_docs = 0
    chunked_docs = []

    logging.info(f"Total documents to process: {total_docs}")

    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)

    # We'll define a helper to save CSV
    def save_csv_and_metrics():
        """
        Saves the current DataFrame to CSV, then runs evaluation metrics.
        """
        # 4) Save results
        sanitized_model_name = config.MODEL_NAME.replace("/", "-")
        sanitized_model_name = re.sub(r'[<>:"/\\|?*]', '-', sanitized_model_name)

        output_csv = f"{config.QUESTION_FILE}_{sanitized_model_name}.csv"
        output_path = os.path.join(output_dir, output_csv)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved LLM answers to {output_path}")

        # 5) Evaluate metrics
        evaluator = BenchmarkEvaluator(results_dir=config.OUTPUT_PATH, metrics_dir=config.METRICS_PATH)
        evaluator.evaluate_all()
        logging.info(f"Saved metrics to {evaluator.metrics_dir}")

    try:
        # NEW OR CHANGED: We wrap the main doc-loop in a try/except
        for doc_id in all_doc_ids:
            processed_docs += 1
            logging.info(f"Processing document {processed_docs}/{total_docs} (doc_id={doc_id})...")

            group_indices = grouped.groups[doc_id]
            overall_indices_list = list(group_indices)

            question_batch_length = 50
            doc_chunks = load_document_text(str(doc_id))  # list of text chunks

            # If doc text is empty, mark all as 'No doc text'
            if not doc_chunks:
                logging.warning(f"Document {doc_id} is empty. Setting llm_response='No doc text'.")
                for idx in overall_indices_list:
                    df.at[idx, "llm_response"] = "No doc text"
                # Save partial results after finishing each doc
                save_csv_and_metrics()
                continue

            # We process the doc's questions in sub-batches
            for q_start in range(0, len(overall_indices_list), question_batch_length):
                q_indices_list = overall_indices_list[q_start: q_start + question_batch_length]
                questions = df.loc[q_indices_list, "question"].tolist()
                num_questions = len(questions)
                logging.info(f"Processing {num_questions} questions for doc_id={doc_id}...")

                # If there's only 1 chunk, process it normally
                if len(doc_chunks) == 1:
                    single_chunk_text = doc_chunks[0]
                    if config.context_chat:
                        for i, row_idx in enumerate(q_indices_list, start=1):
                            question_text = df.at[row_idx, "question"]
                            log_msg = f"[doc={doc_id} chunk=1 question_index={i}]"
                            messages = build_prompt_single(single_chunk_text, question_text, i)
                            parsed_answers = get_llm_json_response(llm, messages, 1, extra_log_info=log_msg)
                            df.at[row_idx, "llm_response"] = parsed_answers[1]
                    else:
                        log_msg = f"[doc={doc_id} chunk=1 batch_mode]"
                        messages = build_prompt_batch(single_chunk_text, questions)
                        parsed_answers = get_llm_json_response(llm, messages, num_questions, extra_log_info=log_msg)
                        for i, row_idx in enumerate(q_indices_list, start=1):
                            df.at[row_idx, "llm_response"] = parsed_answers[i]

                else:
                    # Document required chunking, add to tracking
                    chunked_docs.append(doc_id)

                    # If multiple chunks
                    if config.context_chat:
                        # Each question is separate across all chunks
                        for i, row_idx in enumerate(q_indices_list, start=1):
                            question_text = df.at[row_idx, "question"]
                            partial_responses = []

                            for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                                log_msg = f"[doc={doc_id} chunk={c_idx} question_index={i}]"
                                messages_chunk = build_prompt_single(chunk_text, question_text, i)
                                chunk_parsed_answers = get_llm_json_response(
                                    llm, messages_chunk, 1, extra_log_info=log_msg
                                )

                                if "LLM parse error" in chunk_parsed_answers[1]:
                                    partial_responses.append('{"answers":[{"question_index":1,"answer":"LLM parse error"}]}')
                                else:
                                    partial_json_str = json.dumps({
                                        "answers": [
                                            {"question_index": 1, "answer": chunk_parsed_answers[1]}
                                        ]
                                    })
                                    partial_responses.append(partial_json_str)

                            combine_msg = f"[doc={doc_id} combine question_index={i}]"
                            combine_prompt = build_prompt_combine_answers(partial_responses, [question_text])
                            combined_final = get_llm_json_response(llm, combine_prompt, 1, extra_log_info=combine_msg)
                            df.at[row_idx, "llm_response"] = combined_final[1]

                    else:
                        # Batch mode across multiple chunks
                        partial_responses = []
                        for c_idx, chunk_text in enumerate(doc_chunks, start=1):
                            log_msg = f"[doc={doc_id} chunk={c_idx} batch_mode]"
                            messages_chunk = build_prompt_batch(chunk_text, questions)
                            chunk_parsed_answers = get_llm_json_response(
                                llm, messages_chunk, num_questions, extra_log_info=log_msg
                            )

                            if any(ans == "LLM parse error" for ans in chunk_parsed_answers.values()):
                                fake_json = {
                                    "answers": [
                                        {"question_index": i, "answer": "LLM parse error"}
                                        for i in range(1, num_questions + 1)
                                    ]
                                }
                                partial_responses.append(json.dumps(fake_json))
                            else:
                                partial_json = {"answers": []}
                                for q_idx in range(1, num_questions + 1):
                                    partial_json["answers"].append(
                                        {"question_index": q_idx, "answer": chunk_parsed_answers[q_idx]}
                                    )
                                partial_responses.append(json.dumps(partial_json))

                        combine_msg = f"[doc={doc_id} combine batch_mode]"
                        combine_prompt = build_prompt_combine_answers(partial_responses, questions)
                        combined_output = get_llm_json_response(llm, combine_prompt, num_questions, extra_log_info=combine_msg)
                        for i, row_idx in enumerate(q_indices_list, start=1):
                            df.at[row_idx, "llm_response"] = combined_output[i]

            logging.info(f"Completed processing document {processed_docs}/{total_docs} (doc_id={doc_id}).")
            # NEW OR CHANGED: partial saving after each doc
            save_csv_and_metrics()

    except KeyboardInterrupt:
        # NEW OR CHANGED: If the user presses Ctrl+C or otherwise interrupts
        logging.warning("Code terminated by user. Marking unprocessed documents with 'code terminated'...")

        # Mark all documents not processed yet as "code terminated"
        # processed_docs is the count of docs we already did
        unprocessed_docs = all_doc_ids[processed_docs:]  # The ones we haven't started
        for udoc_id in unprocessed_docs:
            indices_ = grouped.groups[udoc_id]
            for idx in indices_:
                if df.at[idx, "llm_response"] == "":
                    df.at[idx, "llm_response"] = "code terminated"

        # Save final partial results
        save_csv_and_metrics()
        logging.warning("Partial results saved. Exiting now.")
        sys.exit(1)

    except Exception as e:
        # If there's an unexpected exception, log it
        logging.error(f"Unexpected top-level error: {e}", exc_info=True)

        # Mark all documents not processed as "code terminated"
        unprocessed_docs = all_doc_ids[processed_docs:]
        for udoc_id in unprocessed_docs:
            indices_ = grouped.groups[udoc_id]
            for idx in indices_:
                if df.at[idx, "llm_response"] == "":
                    df.at[idx, "llm_response"] = "code terminated"

        # Save partial results
        save_csv_and_metrics()
        logging.warning("Partial results saved. Exiting due to fatal error.")
        sys.exit(1)

    # If we complete everything without interruption:
    logging.info(f"Processing complete: {processed_docs}/{total_docs} documents processed successfully.")

    # 6) Summarize chunked docs
    if chunked_docs:
        unique_chunked = list(set(chunked_docs))
        logging.info(f"Documents that required chunking: {len(unique_chunked)}")
        logging.info(f"Chunked Document IDs: {unique_chunked}")
        print("\nDocuments that required chunking:", unique_chunked)
    else:
        logging.info("No documents required chunking.")

if __name__ == "__main__":
    main()
