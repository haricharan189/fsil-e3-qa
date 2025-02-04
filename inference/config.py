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

QUESTION_FILE = "L1"  # e.g., name of CSV (L1.csv) with columns (document_number, question, answer, etc.)
JSON_FILE     = "semi_cleaned_docs.json"  # JSON structure: [ { "id": "4", "data": { "html": "<html>...</html>" }}, ...]

# ------------------------------------------------------------------------------
# LLM Provider Settings
# ------------------------------------------------------------------------------
LLM_PROVIDER = "Custom" #(ANTHROPIC, MISTRAL, GOOGLE) # set this to "Custom" if you want to use own LLM based on server or "OPENLLM_LOCAL" if local laptop llm
MODEL_NAME   = "chatgpt-4o-latest"  # or "gpt-4"
TEMPERATURE  = 0.1

# We want to set the maximum tokens the LLM can generate as output
max_tokens_generation = 2000

# The overall max token context for your LLM. If you only have 8k or 32k, set accordingly.
max_token = 128000

# ------------------------------------------------------------------------------
# Retry Settings
# ------------------------------------------------------------------------------
NUM_RETRIES = 2   # How many times to retry a failing LLM call

# ------------------------------------------------------------------------------
# Single vs. batch question approach
# ------------------------------------------------------------------------------
# - True  => For each question, we send doc text + single question (fresh prompt each time)
# - False => For each doc, send doc text + ALL questions in one prompt
context_chat = False

# ------------------------------------------------------------------------------
# Other
# ------------------------------------------------------------------------------
# Character limit to avoid excessive context or 1MB limit
MAX_CHAR_FOR_SYSTEM = 550000
