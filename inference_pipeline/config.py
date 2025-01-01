"""
Changes:
  - Added directory paths reflecting your existing structure.
  - Added toggles (ENABLE_BLEU_EVAL, ENABLE_LLM_EVAL, ENABLE_HUMAN_FEEDBACK, etc.)
  - Moved all references for JSON file, document numbers, LLM settings, etc. here.
"""


"""
Changes:
- Added toggles for BLEU, Cosine, BERTScore, LLM-based correctness, and human feedback.
- Changed code so you can easily switch them on/off.
"""

import os

################################
# Base Directory
################################
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

################################
# File Paths
################################
JSON_FILE_PATH = os.path.join(BASE_DIR, "0to11_Arnav.json")  
EXTRACTED_CONTENT_DIR = os.path.join(BASE_DIR, "extracted_content")
GROUND_TRUTH_DIR      = os.path.join(BASE_DIR, "ground_truth")
QUERIES_DIR           = os.path.join(BASE_DIR, "queries")
DATAFRAME_DIR         = os.path.join(BASE_DIR, "dataframe")
DATAFRAME_FINAL_DIR   = os.path.join(BASE_DIR, "dataframe_final")
DATAFRAME_RESULTS_DIR = os.path.join(BASE_DIR, "dataframe_results")

################################
# Pipeline Parameters
################################
DOCUMENT_NUMBERS       = [99, 102]  # Example docs
QUESTIONS_PER_DOCUMENT = 3

################################
# LLM Inference Settings
################################
LLM_API_KEY    = "your api key"
LLM_MODEL      = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
LLM_API_URL    = "https://api.together.xyz/v1/chat/completions"

TEMPERATURE         = 0.7
FREQUENCY_PENALTY   = 0
PRESENCE_PENALTY    = 0
INITIAL_DELAY       = 1
MAX_DELAY           = 30
MAX_RETRIES         = 3

################################
# Evaluation LLM (for correctness checking)
################################
EVAL_LLM_API_KEY = "your api key"  
EVAL_LLM_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
EVAL_LLM_API_URL = "https://api.together.xyz/v1/chat/completions"
EVAL_TEMPERATURE = 0.7  # No randomness for correctness checking

################################
# Evaluation Toggles
################################
ENABLE_BLEU_EVAL       = True   # If True, compute BLEU
ENABLE_LLM_EVAL        = True   # If True, do LLM-based correctness checks
ENABLE_HUMAN_FEEDBACK  = False  # If True, prompt user for feedback in console

# Additional toggles for Cosine & BERTScore
ENABLE_COSINE_EVAL     = True
ENABLE_BERTSCORE_EVAL  = True
