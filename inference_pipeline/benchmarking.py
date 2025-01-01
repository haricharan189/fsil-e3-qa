"""
benchmarking.py

Changes:
 - We respect config.ENABLE_COSINE_EVAL and config.ENABLE_BERTSCORE_EVAL to decide if we compute Cosine or BERTScore.
 - The function now returns 5 items: (doc_scores, overall_bleu, overall_cosine, overall_bert, overall_corr).
 - We have extra print/log statements and safety checks.
 - The 'too many values to unpack' error is solved by matching main.py properly.
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
import requests
import json

import config
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt_tab', quiet=True)

# BERTScore
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

# Sentence Transformers for Cosine Similarity
try:
    from sentence_transformers import SentenceTransformer
    import torch
    COSINE_AVAILABLE = True
except ImportError:
    COSINE_AVAILABLE = False


class BenchmarkEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Inference results
        self.final_dir   = config.DATAFRAME_FINAL_DIR
        # Where to save "benchmark_results.csv"
        self.results_dir = config.DATAFRAME_RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        
        # LLM for evaluation (correctness checking)
        self.eval_model   = config.EVAL_LLM_MODEL
        self.eval_api_key = config.EVAL_LLM_API_KEY
        self.eval_api_url = config.EVAL_LLM_API_URL
        self.eval_temp    = config.EVAL_TEMPERATURE

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.eval_api_key}"
        }

        # Load or init models if toggles are True
        self.cosine_model = None
        if config.ENABLE_COSINE_EVAL and COSINE_AVAILABLE:
            self.cosine_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Loaded SentenceTransformer for Cosine Similarity (all-MiniLM-L6-v2).")

        self.logger.info(f"BERTScore installed? {BERTSCORE_AVAILABLE}, toggle={config.ENABLE_BERTSCORE_EVAL}")
        self.logger.info(f"Cosine   installed? {COSINE_AVAILABLE}, toggle={config.ENABLE_COSINE_EVAL}")

    def evaluate_documents(self):
        """
        For each 'results_{doc_num}.csv' in self.final_dir:
          - compute BLEU if ENABLE_BLEU_EVAL
          - compute Cosine if ENABLE_COSINE_EVAL
          - compute BERTScore if ENABLE_BERTSCORE_EVAL
          - do LLM correctness if ENABLE_LLM_EVAL
        Returns: (doc_scores, overall_bleu, overall_cosine, overall_bert, overall_corr)
        """
        result_files = glob.glob(os.path.join(self.final_dir, "results_*.csv"))
        if not result_files:
            self.logger.warning(f"No CSV files found in {self.final_dir} => results_*.csv not present.")
            # Return empty
            return {}, 0, 0, 0, 0

        doc_scores = {}

        all_bleu_scores = []
        all_cosine_vals = []
        all_bertscores  = []
        all_correctness = []

        for file_path in result_files:
            df = pd.read_csv(file_path)
            doc_num = os.path.basename(file_path).replace("results_", "").replace(".csv", "")

            self.logger.info(f"Evaluating doc {doc_num} with {len(df)} questions...")

            # 1) BLEU
            bleu_scores = []
            if not df.empty and config.ENABLE_BLEU_EVAL:
                for idx, row in df.iterrows():
                    gt      = str(row['ground_truth'])
                    llm_ans = str(row['llm_response'])
                    ref     = [nltk.word_tokenize(gt.lower())]
                    cand    = nltk.word_tokenize(llm_ans.lower())
                    smoothie= SmoothingFunction().method4
                    bleu    = sentence_bleu(ref, cand, smoothing_function=smoothie)
                    bleu_scores.append(bleu)
            else:
                bleu_scores = [0]*len(df)
            mean_bleu = float(np.mean(bleu_scores)) if len(bleu_scores) else 0
            all_bleu_scores.extend(bleu_scores)

            # 2) Cosine Similarity
            cos_vals = []
            if (not df.empty 
                and config.ENABLE_COSINE_EVAL 
                and COSINE_AVAILABLE 
                and self.cosine_model is not None):
                for idx, row in df.iterrows():
                    gt      = str(row['ground_truth'])
                    llm_ans = str(row['llm_response'])
                    embeddings = self.cosine_model.encode([gt, llm_ans], convert_to_tensor=True)
                    cos_sim    = float(torch.nn.functional.cosine_similarity(
                        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
                    ))
                    cos_vals.append(cos_sim)
            else:
                cos_vals = [0]*len(df)
            mean_cosine = float(np.mean(cos_vals)) if cos_vals else 0
            all_cosine_vals.extend(cos_vals)

            # 3) BERTScore
            bert_scores = []
            if (not df.empty 
                and config.ENABLE_BERTSCORE_EVAL 
                and BERTSCORE_AVAILABLE):
                references = df['ground_truth'].tolist()
                cands      = df['llm_response'].tolist()
                # Using default lang="en"
                try:
                    P, R, F1 = bert_score_fn(cands, references, lang="en", verbose=False)
                    bert_scores = list(F1.numpy())
                except Exception as be:
                    self.logger.error(f"BERTScore error for doc {doc_num}: {be}")
                    bert_scores = [0]*len(df)
            else:
                bert_scores = [0]*len(df)
            mean_bertscore = float(np.mean(bert_scores)) if bert_scores else 0
            all_bertscores.extend(bert_scores)

            # 4) LLM-based correctness
            correctness_flags = []
            if not df.empty and config.ENABLE_LLM_EVAL:
                correctness_flags = self.evaluate_document_llm(doc_num, df)
            else:
                correctness_flags = [0]*len(df)
            mean_correctness = float(np.mean(correctness_flags)) if correctness_flags else 0
            all_correctness.extend(correctness_flags)

            # (Optional) Human feedback
            if config.ENABLE_HUMAN_FEEDBACK and not df.empty:
                for idx, row in df.iterrows():
                    user_fb = input(
                        f"[HUMAN FEEDBACK] Doc {doc_num} (Q{idx+1}/{len(df)})\n"
                        f" GT: {row['ground_truth']}\n"
                        f" LLM: {row['llm_response']}\n"
                        f"Is it correct? (y/n) "
                    )
                    # Could store user_fb somewhere, e.g. row['human_feedback'] = user_fb

            # Collect doc-level metrics
            doc_scores[doc_num] = {
                "mean_bleu":        mean_bleu,
                "mean_cosine":      mean_cosine,
                "mean_bertscore":   mean_bertscore,
                "correctness_rate": mean_correctness,
                "num_questions":    len(df)
            }

        # Overall aggregates
        overall_bleu = float(np.mean(all_bleu_scores)) if all_bleu_scores else 0
        overall_cosine = float(np.mean(all_cosine_vals)) if all_cosine_vals else 0
        overall_bert = float(np.mean(all_bertscores)) if all_bertscores else 0
        overall_corr = float(np.mean(all_correctness)) if all_correctness else 0

        return doc_scores, overall_bleu, overall_cosine, overall_bert, overall_corr

    def evaluate_document_llm(self, doc_num, df):
        """
        One LLM call per doc => returns a list of 0/1 correctness flags
        We log token usage if 'usage' is in the response.
        """
        context = self._get_document_content(doc_num)
        # Build one big prompt
        questions_block = []
        for idx, row in df.iterrows():
            q  = row['question']
            gt = row['ground_truth']
            ans= row['llm_response']
            questions_block.append(
                f"Q{idx+1}:\n"
                f"  Question: {q}\n"
                f"  Ground Truth: {gt}\n"
                f"  LLM Answer: {ans}\n"
                "  Evaluate correctness (1=correct or close, 0=incorrect)\n"
            )
        joined = "\n".join(questions_block)
        print(f"this is the message sent to llm for doc {doc_num}: {joined}")
        user_prompt = f"""you are a profession evaluator, your task is to for each Document ID: {doc_num}

Context (if any):
{""}
evaluate the the answer generated by The LLM, if they are close enough and correct then return 1 else 0, nothing else.
Questions & Answers:
{joined}

Respond with one line per Q, '1' or '0'.
"""

        payload = {
            "model":       self.eval_model,
            "temperature": self.eval_temp,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a correctness evaluator. Provide '1' or '0' per question."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        try:
            resp = requests.post(self.eval_api_url, json=payload, headers=self.headers)
            self.logger.info(f"[Doc {doc_num}] LLM-based eval => status: {resp.status_code}")

            if resp.status_code == 429:
                self.logger.error("Eval LLM: Rate limit reached.")
                return [0]*len(df)
            if resp.status_code != 200:
                self.logger.error(f"Eval LLM error: {resp.status_code} => {resp.text}")
                raise requests.exceptions.RequestException(f"Eval LLM returned {resp.status_code}")

            data = resp.json()

            # Log usage/tokens if available
            usage_info = data.get("usage", {})
            total_tokens = usage_info.get("total_tokens", 0)
            self.logger.info(f"[Doc {doc_num}] LLM-based eval used {total_tokens} tokens.")

            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if not reply:
                self.logger.error("Empty response from eval LLM.")
                return [0]*len(df)

            self.logger.info(f"LLM response:\n{reply}")

            lines = reply.splitlines()
            flags = []
            for line in lines:
                line_str = line.strip()
                if line_str.startswith('1'):
                    flags.append(1)
                elif line_str.startswith('0'):
                    flags.append(0)
                else:
                    flags.append(0)

            # pad/truncate as needed
            if len(flags) < len(df):
                flags += [0]*(len(df)-len(flags))
            flags = flags[:len(df)]
            return flags

        except Exception as e:
            self.logger.error(f"LLM-based evaluation failed for doc {doc_num}: {str(e)}")
            return [0]*len(df)

    def _get_document_content(self, doc_num):
        """
        Replicates logic from InferencePipeline, returning doc's HTML or empty if not found.
        """
        try:
            doc_id = int(doc_num)
        except ValueError:
            self.logger.error(f"Doc {doc_num} not an integer => cannot retrieve context.")
            return ""

        try:
            with open(config.JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            for entry in all_data:
                if entry.get("id") == doc_id:
                    html_content = entry.get("data", {}).get("html", "")
                    if html_content:
                        self.logger.info(f"Document {doc_id}: context found.")
                        return html_content
                    else:
                        self.logger.warning(f"Document {doc_id}: no HTML content.")
                        return ""
            self.logger.error(f"Document {doc_id} not found in JSON.")
            return ""
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {config.JSON_FILE_PATH}")
            return ""
        except Exception as ex:
            self.logger.error(f"Error reading JSON for doc {doc_num}: {ex}")
            return ""

def main():
    evaluator = BenchmarkEvaluator()
    (doc_scores, overall_bleu, overall_cosine, overall_bert, overall_corr) = evaluator.evaluate_documents()
    
    # Build DataFrame
    results_df = pd.DataFrame.from_dict(doc_scores, orient='index').reset_index(drop=True)
    # Optionally parse doc_num from doc_scores keys if needed
    # results_df['document_number'] = results_df.index
    # We'll just keep the current approach from doc_scores

    results_df['overall_bleu']   = overall_bleu
    results_df['overall_cosine'] = overall_cosine
    results_df['overall_bert']   = overall_bert
    results_df['overall_correctness'] = overall_corr

    # Save
    out_csv = os.path.join(config.DATAFRAME_RESULTS_DIR, "benchmark_results.csv")
    results_df.to_csv(out_csv, index=False)

    print("\nDocument Scores:")
    for idx, row in results_df.iterrows():
        # If you stored doc_num in the dictionary key or have it in columns, print it
        # doc_id = row.get('document_number', 'unknown')
        print(
            f"Row {idx}: "
            f"BLEU={row['mean_bleu']:.4f}, "
            f"Cosine={row['mean_cosine']:.4f}, "
            f"BERTScore={row['mean_bertscore']:.4f}, "
            f"correctness={row['correctness_rate']:.2f}, "
            f"#Q={row['num_questions']}"
        )
    print(f"\nOverall BLEU: {overall_bleu:.4f}")
    print(f"Overall Cosine: {overall_cosine:.4f}")
    print(f"Overall BERTScore: {overall_bert:.4f}")
    print(f"Overall correctness: {overall_corr:.2f}")

if __name__ == "__main__":
    main()
