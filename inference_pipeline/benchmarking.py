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

        self.final_dir       = config.DATAFRAME_FINAL_DIR
        self.results_dir     = config.DATAFRAME_RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
        # LLM for evaluation
        self.eval_api_key   = config.EVAL_LLM_API_KEY
        self.eval_model     = config.EVAL_LLM_MODEL
        self.eval_api_url   = config.EVAL_LLM_API_URL
        self.eval_temp      = config.EVAL_TEMPERATURE
        
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.eval_api_key}"
        }

    def evaluate_documents(self):
        result_files = glob.glob(os.path.join(self.final_dir, "results_*.csv"))
        
        doc_scores = {}
        all_bleu_scores = []
        all_correctness = []
        
        for file_path in result_files:
            df = pd.read_csv(file_path)
            doc_num = os.path.basename(file_path).replace("results_", "").replace(".csv", "")

            bleu_scores = []
            correctness_flags = []
            
            for idx, row in df.iterrows():
                ground_truth = str(row['ground_truth'])
                llm_response = str(row['llm_response'])

                # BLEU EVAL?
                bleu_score = 0
                if config.ENABLE_BLEU_EVAL:
                    reference = [nltk.word_tokenize(ground_truth.lower())]
                    candidate = nltk.word_tokenize(llm_response.lower())
                    smoothie  = SmoothingFunction().method4
                    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                bleu_scores.append(bleu_score)
                
                # LLM EVAL?
                correctness_val = 0
                if config.ENABLE_LLM_EVAL:
                    correctness_val = self.llm_based_evaluation(
                        question=row['question'],
                        ground_truth=ground_truth,
                        llm_answer=llm_response,
                        doc_num=doc_num
                    )
                correctness_flags.append(correctness_val)

                # HUMAN FEEDBACK? (placeholder)
                if config.ENABLE_HUMAN_FEEDBACK:
                    user_fb = input(
                        f"Doc {doc_num}, Q: {row['question']}\nGT: {ground_truth}\nLLM: {llm_response}\nIs it correct? (y/n) "
                    )
                    # In real scenario, you might do something with user_fb...
            
            avg_bleu = np.mean(bleu_scores) if len(bleu_scores) > 0 else 0
            avg_correctness = np.mean(correctness_flags) if len(correctness_flags) > 0 else 0

            doc_scores[doc_num] = {
                "mean_bleu": avg_bleu,
                "correctness_rate": avg_correctness,
                "num_questions": len(df)
            }
            all_bleu_scores.extend(bleu_scores)
            all_correctness.extend(correctness_flags)

        overall_bleu = np.mean(all_bleu_scores) if all_bleu_scores else 0
        overall_correctness = np.mean(all_correctness) if all_correctness else 0
        
        return doc_scores, overall_bleu, overall_correctness

    def llm_based_evaluation(self, question, ground_truth, llm_answer, doc_num):
        """
        Simple approach: ask the eval LLM if the answer is correct or close enough:
          1 => correct, 0 => incorrect
        """
        content= InferencePipeline.get_document_content(self, doc_num)
        # Simple prompt
        prompt_content = f"""
        You are an evaluator. Check if the LLM's answer is correct or close enough.
        Document: {doc_num}
        Context: {content}
        Question: {question}
        Ground Truth: {ground_truth}
        LLM Answer: {llm_answer}

        Respond with '1' if correct, '0' if incorrect.
        """
        payload = {
            "model": self.eval_model,
            "temperature": self.eval_temp,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        }
        try:
            resp = requests.post(self.eval_api_url, json=payload, headers=self.headers)
            resp.raise_for_status()
            data  = resp.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if reply.startswith('1'):
                return 1
            return 0
        except Exception as e:
            self.logger.error(f"LLM-based evaluation failed for doc {doc_num}: {str(e)}")
            return 0

def main():
    evaluator = BenchmarkEvaluator()
    doc_scores, overall_bleu, overall_corr = evaluator.evaluate_documents()
    
    # Build DataFrame
    results_df = pd.DataFrame.from_dict(doc_scores, orient='index')
    results_df['document_number']      = results_df.index
    results_df['overall_bleu']         = overall_bleu
    results_df['overall_correctness']  = overall_corr

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
