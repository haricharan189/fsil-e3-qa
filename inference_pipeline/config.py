"""
Changes:
  - Added directory paths reflecting your existing structure.
  - Added toggles (ENABLE_BLEU_EVAL, ENABLE_LLM_EVAL, ENABLE_HUMAN_FEEDBACK, etc.)
  - Moved all references for JSON file, document numbers, LLM settings, etc. here.
"""

import os

################################
# Base Directory (relative)
################################
# If this config.py is inside "inference_pipeline" directory, 
# we can build paths relative to the parent directory.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

################################
# Pipeline / File Paths
################################
JSON_FILE_PATH = os.path.join(BASE_DIR, "0to11_Arnav.json")  
# ^ Example JSON; adjust to your actual file (like "48 to 53_hari.json", etc.)

EXTRACTED_CONTENT_DIR = os.path.join(BASE_DIR, "extracted_content")
GROUND_TRUTH_DIR      = os.path.join(BASE_DIR, "ground_truth")
QUERIES_DIR           = os.path.join(BASE_DIR, "queries")
DATAFRAME_DIR         = os.path.join(BASE_DIR, "dataframe")
DATAFRAME_FINAL_DIR   = os.path.join(BASE_DIR, "dataframe_final")
DATAFRAME_RESULTS_DIR = os.path.join(BASE_DIR, "dataframe_results")  # final benchmark CSV

################################
# Pipeline Parameters
################################
DOCUMENT_NUMBERS       = [99,102]  # Example docs
QUESTIONS_PER_DOCUMENT = 4

################################
# LLM Inference Settings
################################
LLM_API_KEY    = "2bdb92211a18a3e781668866d677e9ef6e27097127b6b8f08d4420570ab38024"  
LLM_MODEL      = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
LLM_API_URL    = "https://api.together.xyz/v1/chat/completions"

# Hyperparams
TEMPERATURE         = 0.7
FREQUENCY_PENALTY   = 0
PRESENCE_PENALTY    = 0
INITIAL_DELAY       = 1
MAX_DELAY           = 30
MAX_RETRIES         = 3

################################
# Evaluation LLM (for correctness checking)
################################
EVAL_LLM_API_KEY = "2bdb92211a18a3e781668866d677e9ef6e27097127b6b8f08d4420570ab38024"  # can be same as LLM_API_KEY or different
EVAL_LLM_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
EVAL_LLM_API_URL = "https://api.together.xyz/v1/chat/completions"
EVAL_TEMPERATURE = 0.0  # to reduce randomness

################################
# Evaluation Toggles
################################
ENABLE_BLEU_EVAL       = True   # If True, compute BLEU
ENABLE_LLM_EVAL        = True   # If True, do LLM-based correctness checks
ENABLE_HUMAN_FEEDBACK  = False  # If True, prompt user for feedback in consoleA