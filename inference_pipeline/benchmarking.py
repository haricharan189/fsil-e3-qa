"""
Changes:
  - Evaluates results_{doc_num}.csv from dataframe_final/
  - If ENABLE_BLEU_EVAL is True, compute BLEU via NLTK.
  - If ENABLE_LLM_EVAL is True, do the LLM correctness check.
  - If ENABLE_HUMAN_FEEDBACK is True, ask console input (placeholder).
  - Writes final "benchmark_results.csv" to dataframe_results/.
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
import requests
import json

import config
from inference import InferencePipeline

# For BLEU
import nltk
nltk.data.path.append('/home/arnav/miniconda3/envs/EEE/nltk_data') ## whoever is using this code try running after removing this line, if it doesnt work paste your nltk_data path here

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt_tab', quiet=True)

class BenchmarkEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.final_dir   = config.DATAFRAME_FINAL_DIR   # e.g. "dataframe_final"
        self.results_dir = config.DATAFRAME_RESULTS_DIR # e.g. "dataframe_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # LLM for evaluation
        self.eval_model   = config.EVAL_LLM_MODEL
        self.eval_api_key = config.EVAL_LLM_API_KEY
        self.eval_api_url = config.EVAL_LLM_API_URL
        self.eval_temp    = config.EVAL_TEMPERATURE  # or set a different config param if you like

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.eval_api_key}"
        }

    def evaluate_documents(self):
        """
        Main entry point: 
        - For each 'results_{doc_num}.csv' in self.final_dir
        - Calculate BLEU for each row
        - If LLM-based eval is enabled, call `evaluate_document_llm` once for the entire doc
        - Aggregate correctness results
        Returns doc_scores, overall_bleu, overall_correctness
        """
        result_files = glob.glob(os.path.join(self.final_dir, "results_*.csv"))
        
        doc_scores = {}
        all_bleu_scores = []
        all_correctness = []
        
        for file_path in result_files:
            df = pd.read_csv(file_path)
            doc_num = os.path.basename(file_path).replace("results_", "").replace(".csv", "")

            # 1) Compute BLEU for each row
            bleu_scores = []
            if not df.empty and config.ENABLE_BLEU_EVAL:
                for _, row in df.iterrows():
                    ground_truth = str(row['ground_truth'])
                    llm_response = str(row['llm_response'])
                    reference    = [nltk.word_tokenize(ground_truth.lower())]
                    candidate    = nltk.word_tokenize(llm_response.lower())
                    smoothie     = SmoothingFunction().method4
                    bleu         = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                    bleu_scores.append(bleu)
            else:
                # If BLEU not enabled or no rows
                bleu_scores = [0]*len(df)

            avg_bleu = np.mean(bleu_scores) if len(bleu_scores) > 0 else 0
            all_bleu_scores.extend(bleu_scores)

            # 2) LLM-based correctness check (one call per doc)
            correctness_flags = []
            if not df.empty and config.ENABLE_LLM_EVAL:
                correctness_flags = self.evaluate_document_llm(doc_num, df)
            else:
                correctness_flags = [0]*len(df)

            avg_correctness = np.mean(correctness_flags) if len(correctness_flags) > 0 else 0
            all_correctness.extend(correctness_flags)

            # 3) (Optional) Human feedback
            if config.ENABLE_HUMAN_FEEDBACK and not df.empty:
                for _, row in df.iterrows():
                    user_fb = input(
                        f"Doc {doc_num}\n"
                        f"Q: {row['question']}\n"
                        f"GT: {row['ground_truth']}\n"
                        f"LLM: {row['llm_response']}\n"
                        "Is it correct? (y/n) "
                    )
                    # do something with user_fb if needed

            doc_scores[doc_num] = {
                "mean_bleu": avg_bleu,
                "correctness_rate": avg_correctness,
                "num_questions": len(df)
            }

        overall_bleu        = np.mean(all_bleu_scores) if all_bleu_scores else 0
        overall_correctness = np.mean(all_correctness) if all_correctness else 0
        
        return doc_scores, overall_bleu, overall_correctness

    def evaluate_document_llm(self, doc_num, df):
        """
        Perform a single LLM call to evaluate correctness of each question 
        in df (all rows for the same document).
        Returns a list of 1/0 correctness flags in the same order as df's rows.

        We pass the doc's context ONCE, plus a structured list of Q/GT/LLM. 
        We ask the model to respond with exactly one line per question, '1' or '0'.
        """
        # 1) Get doc context from JSON, similarly to InferencePipeline
        context = self._get_document_content(doc_num)
        
        # 2) Build a single prompt with all Q/GT/LLM
        #    We'll also have a "system" message to ensure model acts as an evaluator
        questions_block = []
        for idx, row in df.iterrows():
            q = row['question']
            gt = row['ground_truth']
            llm_ans = row['llm_response']
            questions_block.append(
                f"Q{idx+1}:\n"
                f"  Question: {q}\n"
                f"  Ground Truth: {gt}\n"
                f"  LLM Answer: {llm_ans}\n"
                f"  Evaluate correctness (1=correct or close enough, 0=incorrect)\n"
            )
        joined_questions = "\n".join(questions_block)

        # We'll keep the prompt minimal to avoid token overflows
        user_prompt = f"""Document ID: {doc_num}


Here are the questions and answers:

{joined_questions}

For each question Q1, Q2, ..., respond with a single line containing '1' if the LLM answer is correct or close to ground truth, 
or '0' if it is incorrect. Output exactly one line per question, no extra text.
"""

        # 3) Build the request in the same shape as inference.py
        payload = {
            "model": self.eval_model,
            "temperature": self.eval_temp,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a correctness evaluator. "
                        "Check each question's LLM answer against the ground truth, "
                        "and output 1 or 0 for each question on separate lines."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        try:
            response = requests.post(self.eval_api_url, json=payload, headers=self.headers)
            self.logger.info(f"Evaluation LLM status: {response.status_code}")

            if response.status_code == 429:
                self.logger.error("Rate limit hit for evaluation LLM.")
                return [0]*len(df)
            if response.status_code != 200:
                self.logger.error(f"Eval API error: {response.status_code} => {response.text}")
                raise requests.exceptions.RequestException(f"Eval API returned {response.status_code}")

            data  = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not reply:
                self.logger.error("Empty response from evaluation LLM.")
                return [0]*len(df)

            # 4) Parse lines => 1 or 0
            lines = reply.splitlines()
            correctness_flags = []
            for line in lines:
                line_str = line.strip()
                if line_str.startswith('1'):
                    correctness_flags.append(1)
                elif line_str.startswith('0'):
                    correctness_flags.append(0)
                else:
                    correctness_flags.append(0)  # fallback if unknown

            # If fewer lines than questions, pad with 0
            if len(correctness_flags) < len(df):
                correctness_flags += [0]*(len(df) - len(correctness_flags))
            # If more lines than needed, truncate
            correctness_flags = correctness_flags[: len(df)]

            return correctness_flags
        
        except Exception as e:
            self.logger.error(f"LLM-based evaluation failed for doc {doc_num}: {str(e)}")
            return [0]*len(df)

    def _get_document_content(self, doc_num):
        """
        A helper method to replicate the logic from InferencePipeline.get_document_content
        but avoids needing to import the entire pipeline.
        """
        try:
            doc_num_int = int(doc_num)  # ensure numeric
        except ValueError:
            self.logger.error(f"Doc {doc_num} is not an integer - can't find in JSON.")
            return ""

        try:
            with open(config.JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                for entry in all_data:
                    if entry.get("id") == doc_num_int:
                        html_content = entry.get("data", {}).get("html", "")
                        return html_content or ""
                self.logger.error(f"Document {doc_num} not found in {config.JSON_FILE_PATH}")
                return ""
        except Exception as e:
            self.logger.error(f"Error reading JSON for doc {doc_num}: {e}")
            return ""

def main():
    evaluator = BenchmarkEvaluator()
    doc_scores, overall_bleu, overall_corr = evaluator.evaluate_documents()
    
    # Build DataFrame
    results_df = pd.DataFrame.from_dict(doc_scores, orient='index')
    results_df['document_number']     = results_df.index
    results_df['overall_bleu']        = overall_bleu
    results_df['overall_correctness'] = overall_corr

    # Save to dataframe_results
    out_csv = os.path.join(config.DATAFRAME_RESULTS_DIR, "benchmark_results.csv")
    results_df.to_csv(out_csv, index=False)

    print("\nDocument Scores:")
    for doc, vals in doc_scores.items():
        print(
            f"Doc {doc}: "
            f"BLEU={vals['mean_bleu']:.4f}, "
            f"correctness={vals['correctness_rate']:.2f}, "
            f"#Q={vals['num_questions']}"
        )
    print(f"\nOverall BLEU: {overall_bleu:.4f}")
    print(f"Overall correctness: {overall_corr:.2f}")

if __name__ == "__main__":
    main()