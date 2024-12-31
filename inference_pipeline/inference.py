"""
Changes:
  - Writes inference results to dataframe_final/
  - Reads from config for LLM details
"""

import pandas as pd
import json
import os
from bs4 import BeautifulSoup
import requests
import time
import logging

import config

class InferencePipeline:
    def __init__(self):
        self.model       = config.LLM_MODEL
        self.api_key     = config.LLM_API_KEY
        self.api_url     = config.LLM_API_URL
        self.headers     = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        self.current_delay      = config.INITIAL_DELAY
        self.max_delay          = config.MAX_DELAY
        self.last_request_time  = 0
        self.consecutive_failures = 0
        self.max_retries        = config.MAX_RETRIES
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.dataframe_dir  = config.DATAFRAME_DIR
        self.final_dir      = config.DATAFRAME_FINAL_DIR
        os.makedirs(self.final_dir, exist_ok=True)
        
        # Load ground truth Q/A CSV
        gt_csv = os.path.join(self.dataframe_dir, "ground_truth_qa.csv")
        self.qa_df = pd.read_csv(gt_csv) if os.path.exists(gt_csv) else pd.DataFrame()

    def _wait_for_rate_limit(self):
        current_time = time.time()
        delta        = current_time - self.last_request_time
        if delta < self.current_delay:
            sleep_time = self.current_delay - delta
            self.logger.info(f"Rate limiting: waiting {sleep_time:.2f}s (current delay: {self.current_delay:.2f}s)")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _increase_delay(self):
        self.consecutive_failures += 1
        self.current_delay = min(self.current_delay * 2, self.max_delay)
        self.logger.warning(f"Increased delay to {self.current_delay:.2f}s after {self.consecutive_failures} failures")

    def _decrease_delay(self):
        if self.consecutive_failures > 0:
            self.consecutive_failures = 0
            self.current_delay = max(self.current_delay * 0.8, 1)
            self.logger.info(f"Decreased delay to {self.current_delay:.2f}s after success")

    def clean_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())
        return text

    def get_document_content(self, document_number):
        try:
            with open(config.JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                for entry in all_data:
                    if entry.get("id") == document_number:
                        html_content = entry.get("data", {}).get("html", "")
                        if html_content:
                            return self.clean_html(html_content)
                        self.logger.warning(f"No HTML content found for doc {document_number}")
                        return None
                self.logger.error(f"Document {document_number} not found in {config.JSON_FILE_PATH}")
                return None
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {config.JSON_FILE_PATH}")
            return None
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON file: {config.JSON_FILE_PATH}")
            return None

    def get_questions_for_document(self, document_number, num_questions=None):
        doc_questions = self.qa_df[self.qa_df['document_number'] == document_number]
        if num_questions:
            return doc_questions.head(num_questions)
        return doc_questions

    def get_llm_answer(self, context, question):
        payload = {
            "model": self.model,
            "temperature": config.TEMPERATURE,
            "frequency_penalty": config.FREQUENCY_PENALTY,
            "presence_penalty": config.PRESENCE_PENALTY,
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                }
            ]
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                resp = requests.post(self.api_url, json=payload, headers=self.headers)
                self.logger.info(f"Response status: {resp.status_code}")
                
                if resp.status_code == 429:
                    self._increase_delay()
                    self.logger.warning("Rate limit exceeded. Retrying...")
                    continue
                if resp.status_code != 200:
                    self.logger.error(f"API error: {resp.status_code} => {resp.text}")
                    raise requests.exceptions.RequestException(f"API returned {resp.status_code}")
                
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not answer:
                    raise ValueError("Empty response from LLM")
                self._decrease_delay()
                return answer.strip()
            
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Attempt {attempt+1}/{self.max_retries} failed: {last_error}")
                self._increase_delay()
                if attempt < self.max_retries - 1:
                    time.sleep(self.current_delay)
        
        error_msg = f"Error after {self.max_retries} attempts. Last error: {last_error}"
        self.logger.error(error_msg)
        return f"[API Error]: {last_error}"

    def process_document(self, document_number, num_questions=None):
        context = self.get_document_content(document_number)
        if not context:
            return None
        
        questions_df = self.get_questions_for_document(document_number, num_questions)
        if questions_df.empty:
            self.logger.warning(f"No ground truth questions found for doc {document_number}")
            return None

        results = []
        total   = len(questions_df)
        for idx, row in questions_df.iterrows():
            self.logger.info(f"Doc {document_number} - Q {idx+1}/{total}")
            llm_answer = self.get_llm_answer(context, row['question'])
            results.append({
                "document_number": document_number,
                "question":        row['question'],
                "ground_truth":    row['answer'],
                "llm_response":    llm_answer,
                "level":           row['level']
            })
            
            # Save interim results every 5
            if (idx+1) % 5 == 0:
                interim_df = pd.DataFrame(results)
                interim_path = os.path.join(self.final_dir, f"interim_results_{document_number}.csv")
                interim_df.to_csv(interim_path, index=False)

        result_df = pd.DataFrame(results)
        output_csv = os.path.join(self.final_dir, f"results_{document_number}.csv")
        result_df.to_csv(output_csv, index=False)
        
        return result_df

def main():
    pipeline = InferencePipeline()
    for doc_id in config.DOCUMENT_NUMBERS:
        pipeline.process_document(document_number=doc_id, num_questions=config.QUESTIONS_PER_DOCUMENT)

if __name__ == "__main__":
    main()
