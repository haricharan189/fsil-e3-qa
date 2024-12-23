import pandas as pd
import os
import glob
from sklearn.metrics import f1_score
import numpy as np

class BenchmarkEvaluator:
    def __init__(self, results_dir="dataframe_final"):
        """
        Initialize the benchmark evaluator.
        """
        self.results_dir = results_dir

    def calculate_f1_score(self, pred, true):
        """
        Calculate simple word-overlap based F1 score.
        """
        if not isinstance(pred, str) or not isinstance(true, str):
            return 0.0
            
        # Convert to lowercase and split into words
        pred_words = set(pred.lower().split())
        true_words = set(true.lower().split())
        
        if not pred_words or not true_words:
            return 0.0
            
        # Calculate precision and recall
        common_words = pred_words.intersection(true_words)
        precision = len(common_words) / len(pred_words)
        recall = len(common_words) / len(true_words)
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1

    def evaluate_documents(self):
        """
        Evaluate all documents and calculate F1 scores.
        """
        # Get all result files (excluding interim results)
        result_files = glob.glob(os.path.join(self.results_dir, "results_*.csv"))
        
        document_scores = {}
        all_f1_scores = []
        
        for file_path in result_files:
            try:
                df = pd.read_csv(file_path)
                doc_number = df['document_number'].iloc[0]
                
                # Calculate F1 scores for each question
                doc_f1_scores = []
                for _, row in df.iterrows():
                    if not row['llm_response'].startswith('API Error:'):
                        f1 = self.calculate_f1_score(row['llm_response'], row['ground_truth'])
                        doc_f1_scores.append(f1)
                        all_f1_scores.append(f1)
                
                # Store document-level metrics
                document_scores[doc_number] = {
                    'mean_f1': np.mean(doc_f1_scores),
                    'num_questions': len(doc_f1_scores)
                }
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Calculate overall F1 score
        overall_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
        
        return document_scores, overall_f1

if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    doc_scores, overall_f1 = evaluator.evaluate_documents()
    
    print("\nDocument-level F1 scores:")
    for doc_num, scores in doc_scores.items():
        print(f"Document {doc_num}:")
        print(f"  Mean F1: {scores['mean_f1']:.4f}")
        print(f"  Questions evaluated: {scores['num_questions']}")
    
    print(f"\nOverall F1 score: {overall_f1:.4f}")
