# config.py

"""
Configuration file for the LLM benchmarking pipeline.
"""

import os

# ------------------------------------------------------------------------------
# Paths and Filenames
# ------------------------------------------------------------------------------
INPUT_PATH = "../data/dataframes/"
OUTPUT_PATH = "../data/results/"
JSON_PATH   = "../data/html_docs/"
METRICS_PATH = "../data/metrics/"
#it is preferred to have sorted question file in increasing order of doc id.
QUESTION_FILE = "L1_test"  #  e.g., name of CSV (L1.csv) with columns (document_number, question, answer, etc.)
JSON_FILE     = "docs_test.json"  # JSON structure: [ { "id": "4", "data": { "html": "<html>...</html>" }}, ...]

# ------------------------------------------------------------------------------
# LLM Provider Settings
# ------------------------------------------------------------------------------
# Examples: "OpenAI", "ANTHROPIC", "MISTRAL", "GOOGLE", "TOGETHER", "Custom"
LLM_PROVIDER = "Custom"
MODEL_NAME   = "/storage/coda1/p-schava6/0/shared/models_Nikita/Llama-3.1-70B-Instruct"   # or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" for Together, etc.
TEMPERATURE  = 0.0

# Maximum tokens to generate in the output
max_tokens_generation = 4000

# The overall max token context for your LLM (8k, 32k, etc. depending on your provider).
max_token = 128000

# ------------------------------------------------------------------------------
# Retry Settings
# ------------------------------------------------------------------------------
NUM_RETRIES = 2   # How many times to retry a failing LLM call

# ------------------------------------------------------------------------------
# Single vs. batch question approach
# ------------------------------------------------------------------------------
# True  => For each question, doc text + single question in separate calls
# False => For each doc, doc text + ALL questions in one call
context_chat = False

# ------------------------------------------------------------------------------
# Other
# ------------------------------------------------------------------------------
# Character limit to avoid context that is too large
MAX_CHAR_FOR_SYSTEM = 550000
