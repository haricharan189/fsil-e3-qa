import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
stemmer = PorterStemmer()

data_dir = "/Users/vidhyakshayakannan/Downloads/cleaned_data"

def preprocess_text(text):
    """Clean and normalize text before comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace(".", "").replace("-", "").replace("?", "").replace("%", "").replace(",", "")
    return " ".join(text.split())  

def compute_f1_score(pred, gold):
    """
    Computes token-level F1-score between predicted and gold answer using stemming
    and preprocessing to normalize text.
    """
    pred = preprocess_text(pred)
    gold = preprocess_text(gold)
    
    if not pred or not gold:
        return 0

    pred_tokens = {stemmer.stem(word) for word in word_tokenize(pred)}
    gold_tokens = {stemmer.stem(word) for word in word_tokenize(gold)}

    true_positives = len(pred_tokens & gold_tokens)
    precision = true_positives / len(pred_tokens) if pred_tokens else 0
    recall = true_positives / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def calculate_edit_distance(self, pred, true):
        """Levenshtein similarity after preprocessing."""
        pred, true = preprocess_text(pred), preprocess_text(true)
        return Levenshtein.distance(pred, true) / max(len(pred), len(true)) if pred and true else 0.0

def calculate_cosine_similarity(self, pred, true):
        """Compute cosine similarity using TF-IDF after preprocessing."""
        pred, true = preprocess_text(pred), preprocess_text(true)
        if not pred or not true:
            return 0.0

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([pred, true])
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        return similarity
pattern = re.compile(r"the\s+([\w\s]+?)\s+and\s+the\s+\1", re.IGNORECASE)

def has_repetition(question):
    """Returns True if the question contains repeated or redundant phrases."""
    if not isinstance(question, str):
        return False
    norm_q = " ".join(question.lower().split())

    # Exact repeated phrase with "and the"
    if re.search(r"the\s+([\w\s]+?)\s+and\s+the\s+\1", norm_q):
        return True

    # Redundancy like "both the lead arranger and the arranger"
    match = re.search(r"both the ([\w\s]+?) and the ([\w\s]+?) in", norm_q)
    if match:
        first = match.group(1).strip()
        second = match.group(2).strip()
        # Remove common qualifiers to compare core roles
        qualifiers = ['lead', 'administrative', 'subsidiary', 'joint']
        first_words = [w for w in first.split() if w not in qualifiers]
        second_words = [w for w in second.split() if w not in qualifiers]
        if first_words == second_words or " ".join(first_words).endswith(" ".join(second_words)):
            return True

    return False

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        if "question" in df.columns:
            repetition_mask = df["question"].apply(has_repetition)
            num_repeated = repetition_mask.sum()
            print(f"{file}: {num_repeated} question(s) detected with repetition.")
            df_clean = df[~repetition_mask].copy()
            df_clean.to_csv(file_path, index=False)
            print(f"Cleaned data stored back to {file}")
        else:
            print(f"File {file} does not have a 'question' column.")

all_dfs = []
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        if {"question", "answer", "llm_response", "num_hops", "num_set_operations", "multiple_answers"}.issubset(df.columns):
            df["F1_Score"] = df.apply(lambda row: compute_f1_score(row["llm_response"], row["answer"]), axis=1)
            df["Source_File"] = file
            all_dfs.append(df)

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
else:
    raise ValueError("No cleaned CSV files with the required columns were found.")

low_f1_threshold = 0.5
low_f1_df = combined_df[combined_df["F1_Score"] < low_f1_threshold]
low_f1_df = low_f1_df[["question", "F1_Score", "num_hops", "num_set_operations", "multiple_answers", "Source_File"]]
low_f1_df = low_f1_df.sort_values(by="F1_Score", ascending=True)
output_path = os.path.join(data_dir, "low_f1_questions_analysis.csv")
low_f1_df.to_csv(output_path, index=False)
print(f"Analysis saved to: {output_path}")
print(low_f1_df.head())

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        if {"question", "answer", "llm_response"}.issubset(df.columns):
            df["F1_Score"] = df.apply(lambda row: compute_f1_score(row["llm_response"], row["answer"]), axis=1)
            df["Edit_Distance_Sim"] = df.apply(lambda row: calculate_edit_distance(None, row["llm_response"], row["answer"]), axis=1)
            df["Cosine_Similarity"] = df.apply(lambda row: calculate_cosine_similarity(None, row["llm_response"], row["answer"]), axis=1)

            df.to_csv(file_path, index=False)
            print(f"Scores added to {file} and saved.")
