import os
import glob
import logging
import pandas as pd
import Levenshtein
import config
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


class BenchmarkEvaluator:
    def __init__(self, results_dir=config.OUTPUT_PATH, metrics_dir=config.METRICS_PATH):
        self.results_dir = results_dir
        self.metrics_dir = metrics_dir
        # Sanitize the model name for file naming
        self.sanitized_model_name = config.MODEL_NAME.replace("/", "-")
        self.sanitized_model_name = re.sub(r'[<>:"/\\|?*]', '-', self.sanitized_model_name)
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
        return 1 - (Levenshtein.distance(pred, true) / max(len(pred), len(true))) if pred and true else 0.0

    def calculate_cosine_similarity(self, pred, true):
        """Compute cosine similarity using TF-IDF after preprocessing."""
        pred, true = preprocess_text(pred), preprocess_text(true)
        if not pred or not true:
            return 0.0

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([pred, true])
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        return similarity

    def evaluate_csv(self, csv_path):
        """Evaluate a single CSV file and save per-question metrics."""
        df = pd.read_csv(csv_path)

        # Check if necessary columns exist
        if "document_number" not in df.columns or "llm_response" not in df.columns or "answer" not in df.columns:
            logging.warning(f"Skipping {csv_path}: Missing necessary columns.")
            return None

        # Compute per-question metrics
        df["f1_score"] = df.apply(lambda row: self.calculate_f1_score(row["llm_response"], row["answer"]), axis=1)
        df["edit_distance"] = df.apply(lambda row: self.calculate_edit_distance(row["llm_response"], row["answer"]), axis=1)
        df["cosine_similarity"] = df.apply(lambda row: self.calculate_cosine_similarity(row["llm_response"], row["answer"]), axis=1)

        # Save question-wise metrics
        doc_metrics_path = os.path.join(self.metrics_dir, f"{config.QUESTION_FILE}_{self.sanitized_model_name}_question_metrics.csv")
        df.to_csv(doc_metrics_path, index=False)
        logging.info(f"Saved question-wise metrics: {doc_metrics_path}")

        return df

    def compute_document_statistics(self, all_dfs):
        """Compute per-document aggregated statistics and overall statistics."""
        document_stats = []
        all_f1_scores, all_edit_distances, all_cosine_similarities = [], [], []

        for df in all_dfs:
            grouped = df.groupby("document_number").agg(
                average_f1_score=("f1_score", "mean"),
                average_edit_distance=("edit_distance", "mean"),
                average_cosine_similarity=("cosine_similarity", "mean"),
                total_questions=("f1_score", "count")
            ).reset_index()

            document_stats.append(grouped)
            all_f1_scores.extend(df["f1_score"].tolist())
            all_edit_distances.extend(df["edit_distance"].tolist())
            all_cosine_similarities.extend(df["cosine_similarity"].tolist())

        # Save per-document statistics
        doc_stats_df = pd.concat(document_stats, ignore_index=True)
        doc_stats_path = os.path.join(self.metrics_dir, f"{config.QUESTION_FILE}_{self.sanitized_model_name}_document_statistics.csv")
        doc_stats_df.to_csv(doc_stats_path, index=False)
        logging.info(f"Saved per-document aggregated metrics: {doc_stats_path}")

        # Compute overall statistics
        overall_stats = {
            "average_f1_score": sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0,
            "average_edit_distance": sum(all_edit_distances) / len(all_edit_distances) if all_edit_distances else 0.0,
            "average_cosine_similarity": sum(all_cosine_similarities) / len(all_cosine_similarities) if all_cosine_similarities else 0.0,
            "total_documents": len(doc_stats_df),
            "total_questions": doc_stats_df["total_questions"].sum() if not doc_stats_df.empty else 0
        }

        overall_stats_df = pd.DataFrame([overall_stats])
        overall_stats_path = os.path.join(self.metrics_dir, f"{config.QUESTION_FILE}_{self.sanitized_model_name}_overall_statistics.csv")
        overall_stats_df.to_csv(overall_stats_path, index=False)
        logging.info(f"Saved overall aggregated metrics: {overall_stats_path}")

    def evaluate_all(self):
        """Evaluate all CSV files in results_dir and create per-document & overall statistics."""
        csv_files = glob.glob(os.path.join(self.results_dir, f"{config.QUESTION_FILE}_{self.sanitized_model_name}.csv"))
        all_dfs = []

        for csv_file in csv_files:
            df = self.evaluate_csv(csv_file)
            if df is not None:
                all_dfs.append(df)

        if all_dfs:
            self.compute_document_statistics(all_dfs)
        else:
            logging.warning("No valid files found for evaluation.")
