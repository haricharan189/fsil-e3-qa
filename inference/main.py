# main.py
import re
import os
import json
import logging
import time
import pandas as pd
from langchain_community.callbacks.manager import get_openai_callback
from bs4 import BeautifulSoup
from metric import BenchmarkEvaluator
import config
from model_loader import BaseModel

logging.basicConfig(level=logging.INFO)

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

def load_document_text(doc_id: str) -> str:
    """
    Load HTML from the JSON file, clean it, optionally truncate if it's too large.
    Returns the final cleaned text.
    """
    json_path = os.path.join(config.JSON_PATH, config.JSON_FILE)
    if not os.path.exists(json_path):
        logging.error(f"JSON file not found: {json_path}")
        return ""

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                if str(entry.get("id")) == str(doc_id):
                    raw_html = entry.get("data", {}).get("html", "")
                    cleaned = clean_html(raw_html)
                    # Truncate if too large:
                    if len(cleaned) > config.MAX_CHAR_FOR_SYSTEM:
                        logging.warning(
                            f"Doc {doc_id} length {len(cleaned)} > {config.MAX_CHAR_FOR_SYSTEM}, truncating..."
                        )
                        cleaned = cleaned[: config.MAX_CHAR_FOR_SYSTEM]
                    logging.info(f"Doc {doc_id} loaded, length={len(cleaned)} chars (not printing full text).")
                    return cleaned
            logging.warning(f"Document {doc_id} not found in {json_path}. Returning empty string.")
            return ""
    except Exception as e:
        logging.error(f"Error reading {json_path}: {e}")
        return ""

def build_prompt_single(document_text: str, question: str, question_index: int) -> list[dict]:
    """
    Single user message combining instructions + doc text + question.
    """
    user_instructions = (
        "You are a helpful assistant. You have the following document text.\n"
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
        "You are a helpful assistant. You have the following document text.",
        "If the answer is not found, say 'Not found'.\n",
        f"Document:\n{document_text}\n",
        "Return your answers in JSON format EXACTLY:\n"
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
            with get_openai_callback() as cb:
                response = llm.invoke(messages)

            # Grab the text from `response.content` if using ChatOpenAI
            raw_output = response.content.strip() if hasattr(response, "content") else str(response).strip()

            if raw_output:
                return raw_output
            else:
                logging.warning(f"Got an empty response from LLM. Attempt {attempt+1}/{config.NUM_RETRIES}. Retrying...")
                time.sleep(1.0)  # small wait before retry

        except Exception as e:
            logging.error(f"LLM call error (attempt {attempt+1}): {e}")
            time.sleep(1.0)  # small wait before retry

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

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    # 3) For each document group, load text & ask the LLM
    for doc_id, group_indices in grouped.groups.items():
        indices_list = list(group_indices)
        doc_text = load_document_text(str(doc_id))
        if not doc_text:
            logging.warning(f"Document {doc_id} is empty. Setting llm_response='No doc text'.")
            for idx in indices_list:
                df.at[idx, "llm_response"] = "No doc text"
            continue

        questions = df.loc[indices_list, "question"].tolist()
        num_questions = len(questions)
        logging.info(f"Processing doc_id={doc_id} with {num_questions} questions...")

        # We track usage for the entire doc in one or multiple calls
        doc_prompt_tokens = 0
        doc_completion_tokens = 0
        doc_cost = 0.0

        if config.context_chat:
            # For each question, create a single prompt
            for i, row_idx in enumerate(indices_list, start=1):
                question_text = df.at[row_idx, "question"]
                logging.info(f"Q{i}/{num_questions} => {question_text}")

                # Build prompt for a single question
                messages = build_prompt_single(doc_text, question_text, i)

                # Call LLM with retries
                raw_output = call_llm_with_retries(llm, messages)
                if not raw_output:
                    df.at[row_idx, "llm_response"] = "LLM error or empty"
                    continue

                # Now count tokens/cost for *this single call*
                try:
                    with get_openai_callback() as cb:
                        # We won't actually re-invoke the LLM for counting usage.
                        # Instead, do a dummy call or skip if you only can measure usage from an actual call.
                        pass
                    # The real usage from that call was already captured by call_llm_with_retries,
                    # but we can't retrieve it post-facto unless we unify the logic.
                    # For demonstration, we just skip or unify calls in the same block.
                except:
                    pass

                # Parse the JSON for 1 question
                parsed_answers = parse_llm_json(raw_output, 1)
                df.at[row_idx, "llm_response"] = parsed_answers[1]

        else:
            # Single prompt for all questions at once
            messages = build_prompt_batch(doc_text, questions)

            raw_output = call_llm_with_retries(llm, messages)
            if not raw_output:
                for i, row_idx in enumerate(indices_list, start=1):
                    df.at[row_idx, "llm_response"] = "LLM error or empty"
                continue

            # We re-call with get_openai_callback() just to measure usage for this single request
            try:
                with get_openai_callback() as cb:
                    # We must do the actual LLM call here to measure usage tokens/cost
                    # But we've already done call_llm_with_retries to get raw_output...
                    # So let's do a second call purely for usage measurement, which is not ideal.
                    # Alternatively, we'd unify usage measurement inside call_llm_with_retries.
                    _ = llm.invoke(messages)

                doc_prompt_tokens     += cb.prompt_tokens
                doc_completion_tokens += cb.completion_tokens
                doc_cost              += cb.total_cost

            except Exception as e:
                logging.error(f"Usage measurement call failed: {e}")

            parsed_answers = parse_llm_json(raw_output, num_questions)
            for i, row_idx in enumerate(indices_list, start=1):
                df.at[row_idx, "llm_response"] = parsed_answers[i]

        # Accumulate doc usage
        total_prompt_tokens     += doc_prompt_tokens
        total_completion_tokens += doc_completion_tokens
        total_cost              += doc_cost

    # 4) Save results
    output_dir = config.OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_csv = f"{config.QUESTION_FILE}_{config.MODEL_NAME}.csv"
    output_path = os.path.join(output_dir, output_csv)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved LLM answers to {output_path}")
    # 5) Summarize total usage/cost
    logging.info(
        f"Total Prompt Tokens: {total_prompt_tokens}, "
        f"Total Completion Tokens: {total_completion_tokens}, "
        f"Total Tokens: {total_prompt_tokens + total_completion_tokens}, "
        f"Estimated Cost (USD): ${total_cost:.4f}"
    )
    # 6) Evaluate metrics
    evaluator = BenchmarkEvaluator(results_dir=config.OUTPUT_PATH, metrics_dir=config.METRICS_PATH)
    evaluator.evaluate_all()
    logging.info(f"Saved metrics to {evaluator.metrics_dir}")
    

if __name__ == "__main__":
    main()
