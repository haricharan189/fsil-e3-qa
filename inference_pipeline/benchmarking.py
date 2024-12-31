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

        self.final_dir   = config.DATAFRAME_FINAL_DIR      # "dataframe_final"
        self.results_dir = config.DATAFRAME_RESULTS_DIR    # "dataframe_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # LLM for evaluation
        self.eval_api_key = config.EVAL_LLM_API_KEY
        self.eval_model   = config.EVAL_LLM_MODEL
        self.eval_api_url = config.EVAL_LLM_API_URL
        self.eval_temp    = config.EVAL_TEMPERATURE
        
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.eval_api_key}"
        }

    def evaluate_documents(self):
        """
        Main entry point: 
        - For each "results_{doc_num}.csv" in self.final_dir
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
            for idx, row in df.iterrows():
                if config.ENABLE_BLEU_EVAL:
                    ground_truth = str(row['ground_truth'])
                    llm_response = str(row['llm_response'])
                    reference    = [nltk.word_tokenize(ground_truth.lower())]
                    candidate    = nltk.word_tokenize(llm_response.lower())
                    smoothie     = SmoothingFunction().method4
                    bleu         = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                else:
                    bleu = 0.0
                bleu_scores.append(bleu)

            # 2) Summarize BLEU
            avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
            all_bleu_scores.extend(bleu_scores)

            # 3) LLM-based correctness check (per document, single call)
            correctness_flags = []
            if config.ENABLE_LLM_EVAL:
                correctness_flags = self.evaluate_document_llm(doc_num, df)
            else:
                # If not enabled, set all to 0 or skip
                correctness_flags = [0]*len(df)

            avg_correctness = np.mean(correctness_flags) if correctness_flags else 0
            all_correctness.extend(correctness_flags)

            # 4) (Optional) Human feedback
            if config.ENABLE_HUMAN_FEEDBACK:
                for idx, row in df.iterrows():
                    user_fb = input(
                        f"Doc {doc_num}, Q: {row['question']}\n"
                        f"GT: {row['ground_truth']}\n"
                        f"LLM: {row['llm_response']}\nIs it correct? (y/n) "
                    )
                    # Save or process user_fb if needed

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
        in df (all rows for a single document).
        Returns a list of 1/0 correctness flags in the same order as df's rows.

        We pass the doc's context ONCE, plus a structured list of Q/GT/LLM. 
        We ask the model to return lines of '1' or '0' for each question.
        """
        # 1) Get the doc context (HTML text) from InferencePipeline code
        #    or from your own method:
        context = self._get_document_content(doc_num)
        
        # 2) Build a single big prompt with the entire context, plus Q/GT/LLM for each row
        # We instruct the model to respond with one line per question, either "1" or "0".
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
                f"  Evaluate correctness (1=correct/close, 0=incorrect)\n"
            )
        joined_questions = "\n".join(questions_block)

        prompt_content = f"""
You are a correctness evaluator.

Document ID: {doc_num}
Context:
{context}

Below are the questions, ground-truth answers, and LLM answers. 
For each question (Q1, Q2, ...), respond with '1' or '0', on separate lines, in the same order. 
No extra text please.

{joined_questions}

Please output exactly one line per question: either "1" or "0".
"""

        # 3) Call the LLM once
        payload = {
            "model":       self.eval_model,
            "temperature": self.eval_temp,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        }
        
        try:
            response = requests.post(self.eval_api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            data  = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # 4) Parse lines from the LLM's reply => a list of 1 or 0
            lines = reply.splitlines()
            correctness_flags = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('1'):
                    correctness_flags.append(1)
                elif line_stripped.startswith('0'):
                    correctness_flags.append(0)
                else:
                    # If unknown format, default to 0 or skip
                    correctness_flags.append(0)
            
            # In case the model returned fewer/more lines than questions
            if len(correctness_flags) < len(df):
                # Fill rest with 0
                correctness_flags.extend([0]*(len(df)-len(correctness_flags)))
            elif len(correctness_flags) > len(df):
                correctness_flags = correctness_flags[:len(df)]
            
            return correctness_flags
        
        except Exception as e:
            self.logger.error(f"LLM-based evaluation failed for doc {doc_num}: {str(e)}")
            # If something fails, default everything to 0
            return [0]*len(df)

    def _get_document_content(self, doc_num):
        """
        A helper method to get the HTML content from your JSON, 
        similarly to InferencePipeline.get_document_content,
        but re-implemented here so we don't need a second pipeline instance.

        If your doc IDs are not in the JSON or doc is missing, return empty string.
        """
        json_path = config.JSON_FILE_PATH
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                for entry in all_data:
                    if entry.get("id") == int(doc_num):
                        html_content = entry.get("data", {}).get("html", "")
                        return html_content  # or do a "clean_html" if desired
                self.logger.error(f"Document {doc_num} not found in {json_path}")
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

    # Print
    print("\nDocument Scores:")
    for doc, vals in doc_scores.items():
        print(f"Doc {doc}: BLEU={vals['mean_bleu']:.4f}, correctness={vals['correctness_rate']:.2f}, #Q={vals['num_questions']}")
    print(f"\nOverall BLEU: {overall_bleu:.4f}")
    print(f"Overall correctness: {overall_corr:.2f}")

if __name__ == "__main__":
    main()