import os
import glob
import logging
import pandas as pd
import Levenshtein
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)


def preprocess_text(text):
    """Clean and normalize text before comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace(".", "").replace("-", "").replace("?", "").replace("%","").replace(",", "") 
    return " ".join(text.split())  # Remove extra spaces


def extract_model_info(filename):
    """Extract level, model name and dimensions from filename."""
    # Example filename: L1_proprietary_gemini-1.5-flash_0_1_0.csv
    pattern = r'(L\d+)_(.+)_(\d+)_(\d+)_(\d+)\.csv'
    match = re.match(pattern, os.path.basename(filename))
    if match:
        level, model_name, multi_answer, num_hops, set_ops = match.groups()
        return {
            'level': level,
            'model_name': model_name,
            'multi_answer': bool(int(multi_answer)),
            'num_hops': int(num_hops),
            'set_ops': int(set_ops)
        }
    return None


def get_output_filename(input_filename):
    """Generate output filename by appending _results.csv to the base name."""
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    return f"{base_name}_results.csv"


class BenchmarkEvaluator:
    def __init__(self, results_dir, metrics_dir):
        self.results_dir = results_dir
        self.metrics_dir = metrics_dir
        os.makedirs(self.metrics_dir, exist_ok=True)

    def calculate_f1_score(self, pred, true):
        """Word overlap F1 score after preprocessing."""
        pred, true = preprocess_text(pred), preprocess_text(true)
        pred_words, true_words = set(pred.split()), set(true.split())

        if not pred_words or not true_words:
            return 0.0

        overlap = pred_words.intersection(true_words)
        precision = len(overlap) / len(pred_words) if pred_words else 0
        recall = len(overlap) / len(true_words) if true_words else 0

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

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

    def calculate_LLM_as_a_judge(self, pred, true):
        system_prompt = (
            "You are an expert evaluator. Your task is to score how well a model's answer matches the ground truth. "
            "Use a scale from 1 to 5, where 5 means 'perfect match', and 1 means 'completely incorrect'. "
            "Respond with only the score as a number (1, 2, 3, 4, or 5)."
        )

    def evaluate_csv(self, csv_path):
        """Evaluate a single CSV file and compute metrics."""
        df = pd.read_csv(csv_path)

        # Check if necessary columns exist
        if "document_number" not in df.columns or "llm_response" not in df.columns or "answer" not in df.columns:
            logging.warning(f"Skipping {csv_path}: Missing necessary columns.")
            return

        # Compute metrics for each question
        df["f1_score"] = df.apply(lambda row: self.calculate_f1_score(row["llm_response"], row["answer"]), axis=1)
        df["edit_distance"] = df.apply(lambda row: self.calculate_edit_distance(row["llm_response"], row["answer"]), axis=1)
        df["cosine_similarity"] = df.apply(lambda row: self.calculate_cosine_similarity(row["llm_response"], row["answer"]), axis=1)

        # Compute document-level statistics
        doc_stats = df.groupby("document_number").agg(
            f1_score=("f1_score", "mean"),
            edit_distance=("edit_distance", "mean"),
            cosine_similarity=("cosine_similarity", "mean"),
            num_questions=("f1_score", "count")
        )

        # Compute overall statistics
        overall_stats = {
            "average_f1_score": df["f1_score"].mean(),
            "average_edit_distance": df["edit_distance"].mean(),
            "average_cosine_similarity": df["cosine_similarity"].mean(),
            "total_documents": len(doc_stats),
            "total_questions": len(df),
            "min_f1_score": df["f1_score"].min(),
            "max_f1_score": df["f1_score"].max(),
            "std_f1_score": df["f1_score"].std(),
            "min_edit_distance": df["edit_distance"].min(),
            "max_edit_distance": df["edit_distance"].max(),
            "std_edit_distance": df["edit_distance"].std(),
            "min_cosine_similarity": df["cosine_similarity"].min(),
            "max_cosine_similarity": df["cosine_similarity"].max(),
            "std_cosine_similarity": df["cosine_similarity"].std()
        }

        # Save overall statistics
        output_filename = get_output_filename(csv_path)
        output_path = os.path.join(self.metrics_dir, output_filename)
        pd.DataFrame([overall_stats]).to_csv(output_path, index=False)
        logging.info(f"Saved metrics for {csv_path} to {output_path}")

    def evaluate_all(self):
        """Evaluate all CSV files in results_dir."""
        csv_files = glob.glob(os.path.join(self.results_dir, "*.csv"))
        
        for csv_file in csv_files:
            logging.info(f"Processing {csv_file}")
            self.evaluate_csv(csv_file)

evaluator = BenchmarkEvaluator(
    results_dir="restructured",
    metrics_dir="metrics"
)
evaluator.evaluate_all()
