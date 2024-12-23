import pandas as pd
import json
import os
from bs4 import BeautifulSoup
import requests
import numpy as np
import time
from datetime import datetime
import logging

class InferencePipeline:
    def __init__(self, api_key, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
                 initial_delay=1):
        """
        Initialize the inference pipeline with adaptive rate limiting.
        
        Args:
            api_key (str): TogetherAI API key
            model (str): Model to use for inference
            initial_delay (float): Initial delay in seconds
        """
        self.model = model
        self.api_key = api_key
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Adaptive rate limiting
        self.current_delay = initial_delay
        self.max_delay = 30  # Maximum delay in seconds
        self.last_request_time = 0
        self.consecutive_failures = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Directories setup
        self.dataframe_dir = "dataframe"
        self.json_dir = "json_files"
        self.final_dir = "dataframe_final"
        os.makedirs(self.final_dir, exist_ok=True)
        
        self.qa_df = pd.read_csv(os.path.join(self.dataframe_dir, "ground_truth_qa.csv"))

    def _wait_for_rate_limit(self):
        """
        Adaptive rate limiting with incrementing delay.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Apply current delay
        if time_since_last_request < self.current_delay:
            sleep_time = self.current_delay - time_since_last_request
            self.logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds (current delay: {self.current_delay:.2f}s)")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _increase_delay(self):
        """
        Increase the delay after a rate limit error.
        """
        self.consecutive_failures += 1
        # Exponential backoff: double the delay up to max_delay
        self.current_delay = min(self.current_delay * 2, self.max_delay)
        self.logger.warning(f"Increased delay to {self.current_delay:.2f}s after {self.consecutive_failures} failures")

    def _decrease_delay(self):
        """
        Gradually decrease the delay after successful requests.
        """
        if self.consecutive_failures > 0:
            self.consecutive_failures = 0
            # Gradually reduce delay but don't go below initial delay
            self.current_delay = max(self.current_delay * 0.8, 1)
            self.logger.info(f"Decreased delay to {self.current_delay:.2f}s after successful request")

    def clean_html(self, html_content):
        """
        Clean HTML content using BeautifulSoup.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean it
        text = soup.get_text(separator=' ')
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def get_document_content(self, document_number):
        """
        Get and clean content for a specific document based on document ID from JSON.
        
        Args:
            document_number (int): Document number to find in the JSON file
        """
        json_path = "48 to 53_hari.json"
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                
                # Find the entry with matching document ID
                for entry in all_data:
                    if entry.get("id") == document_number:
                        # Get the HTML content directly
                        html_content = entry.get("data", {}).get("html", '')
                        if html_content:
                            return self.clean_html(html_content)
                        
                        self.logger.warning(f"No HTML content found for document {document_number}")
                        return None
                
                self.logger.error(f"Document {document_number} not found in JSON file")
                return None
                
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {json_path}")
            return None
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON file: {json_path}")
            return None

    def get_questions_for_document(self, document_number, num_questions=None):
        """
        Get questions for a specific document.
        
        Args:
            document_number (int): Document number
            num_questions (int): Number of questions to return (None for all)
        """
        doc_questions = self.qa_df[self.qa_df['document_number'] == document_number]
        if num_questions:
            return doc_questions.head(num_questions)
        return doc_questions

    def get_llm_answer(self, context, question, max_retries=3):
        """
        Get answer from LLM using TogetherAI API with adaptive retry mechanism.
        """
        payload = {
            "model": self.model,
            "temperature": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "messages": [
                {
                    "role": "user",
                    "content": f"Based on this Context: {context}\n\nAnswer this Question: {question}\nAnswer:"
                }
            ]
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                response = requests.post(self.api_url, json=payload, headers=self.headers)
                
                # Log the response status and headers for debugging
                self.logger.info(f"Response status: {response.status_code}")
                self.logger.info(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 429:  # Too Many Requests
                    self._increase_delay()
                    self.logger.warning(f"Rate limit exceeded. Retrying with {self.current_delay}s delay...")
                    continue
                
                if response.status_code != 200:
                    # Log non-200 responses
                    self.logger.error(f"API returned status {response.status_code}: {response.text}")
                    raise requests.exceptions.RequestException(f"API returned status {response.status_code}")
                    
                response.raise_for_status()
                
                # Try to parse the JSON response
                try:
                    response_data = response.json()
                    answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not answer:
                        raise ValueError("Empty response from API")
                    
                    # Successful request, gradually decrease delay
                    self._decrease_delay()
                    return answer.strip()
                    
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    self.logger.error(f"Failed to parse API response: {str(e)}")
                    self.logger.error(f"Response content: {response.text}")
                    raise
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                self._increase_delay()
                if attempt < max_retries - 1:
                    time.sleep(self.current_delay)
        
        # If we get here, all retries failed
        error_msg = f"Error in LLM response after {max_retries} attempts. Last error: {last_error}"
        self.logger.error(error_msg)
        return f"API Error: {last_error}"  # Return more specific error message

    def process_document(self, document_number, num_questions=None):
        """
        Process a document and get LLM answers for questions.
        """
        # Get document content
        context = self.get_document_content(document_number)
        if not context:
            return None

        # Get questions for the document
        questions_df = self.get_questions_for_document(document_number, num_questions)
        
        # Get LLM answers with progress tracking
        results = []
        total_questions = len(questions_df)
        
        for idx, row in questions_df.iterrows():
            self.logger.info(f"Processing question {idx + 1}/{total_questions} for document {document_number}")
            
            llm_answer = self.get_llm_answer(context, row['question'])
            results.append({
                'document_number': document_number,
                'question': row['question'],
                'ground_truth': row['answer'],
                'llm_response': llm_answer,
                'level': row['level']
            })
            
            # Save intermediate results every 5 questions
            if (idx + 1) % 5 == 0:
                interim_df = pd.DataFrame(results)
                interim_path = os.path.join(self.final_dir, f"interim_results_{document_number}.csv")
                interim_df.to_csv(interim_path, index=False)

        # Create and save final results DataFrame
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.final_dir, f"results_{document_number}.csv")
        results_df.to_csv(output_path, index=False)
        
        return results_df

if __name__ == "__main__":
    api_key = "api_key"
    pipeline = InferencePipeline(api_key, initial_delay=1)
    
    try:
        results = pipeline.process_document(document_number=219, num_questions=4)
        if results is not None:
            print("\nSample of results:")
            print(results.head())
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
