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
os.makedirs(OUTPUT_PATH, exist_ok=True)
JSON_PATH = "../data/json_docs/"
VECTOR_DB_DIR = "../data/vector_store/"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
METRICS_PATH = "../data/metrics/"
os.makedirs(METRICS_PATH, exist_ok=True)
# it is preferred to have sorted question file in increasing order of doc id.
# e.g., name of CSV (L1.csv) with columns (document_number, question, answer, etc.)
QUESTION_FILE = "L1_proprietary"
# JSON structure: [ { "id": "4", "data": { "html": "<html>...</html>" }}, ...]
JSON_FILE = "docs_test.json"

# ------------------------------------------------------------------------------
# LLM Provider Settings
# ------------------------------------------------------------------------------
# Examples: "OpenAI", "ANTHROPIC", "MISTRAL", "GOOGLE", "TOGETHER", "Custom"
# LLM_PROVIDER = "TOGETHER"  # "GOOGLE" "TOGETHER"
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
# LLM_PROVIDER = "GOOGLE"  # "GOOGLE" "TOGETHER"
# MODEL_NAME = "gemini-2.0-flash"
LLM_PROVIDER = "TOGETHER"  # "GOOGLE" "TOGETHER"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
TEMPERATURE = 0.0
# The regime in which the LLM is tested:
# - "FULL" for benchmarking LLMs with entire documents
# - "RAG" for benchmarking LLMs with RAG (retrieval-augmented generation)
# - "GOLD" for benchmarking LLMs with pieces of documents containing the answer
TESTING_REGIME = "RAG"
RAG_TOP_K = 25
RAG_CHUNK_SIZE = 5000
RAG_MODEL = "all-MiniLM-L6-v2"
GOLD_PADDING = 1000

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
WAIT_TIME_ENABLED = True       # Set to True to enable a wait between LLM calls
WAIT_TIME_DURATION = 5

# ------------------------------------------------------------------------------
# Other
# ------------------------------------------------------------------------------
# Character limit to avoid context that is too large
MAX_CHAR_FOR_SYSTEM = 450000
