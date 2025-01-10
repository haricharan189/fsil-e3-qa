import pandas as pd
import os
import glob
from sklearn.metrics import f1_score
import numpy as np
import Levenshtein

class BenchmarkEvaluator:
    def __init__(self, results_dir="dataframe_final"):
        """
        Initialize the benchmark evaluator.
        """
        self.results_dir = results_dir
        # Ensure dataframe_results directory exists
        os.makedirs('dataframe_results', exist_ok=True)

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

    def calculate_token_edit_distance(self, pred, true):
        """
        Calculate token-level Levenshtein distance between prediction and ground truth.
        Returns a normalized score between 0 and 1, where 1 means perfect match.
        """
        if not isinstance(pred, str) or not isinstance(true, str):
            return 0.0
            
        # Convert to lowercase and split into words
        pred_tokens = pred.lower().split()
        true_tokens = true.lower().split()
        
        if not pred_tokens or not true_tokens:
            return 0.0
            
        # Join tokens with a special separator to maintain token boundaries
        pred_str = " ".join(pred_tokens)
        true_str = " ".join(true_tokens)
        
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(pred_str, true_str)
        
        # Normalize by the length of the longer string
        max_len = max(len(pred_str), len(true_str))
        if max_len == 0:
            return 0.0
            
        # Convert distance to similarity score (1 - normalized_distance)
        similarity = 1 - (distance / max_len)
        return similarity

    def evaluate_documents(self):
        """
        Evaluate all documents and calculate F1 scores and token edit distances.
        """
        # Get all result files (excluding interim results)
        result_files = glob.glob(os.path.join(self.results_dir, "results_*.csv"))
        
        document_scores = {}
        all_f1_scores = []
        all_edit_distances = []
        
        for file_path in result_files:
            try:
                df = pd.read_csv(file_path)
                doc_number = df['document_number'].iloc[0]
                
                # Calculate scores for each question
                doc_f1_scores = []
                doc_edit_distances = []
                
                for _, row in df.iterrows():
                    if not row['llm_response'].startswith('API Error:'):
                        f1 = self.calculate_f1_score(row['llm_response'], row['ground_truth'])
                        edit_dist = self.calculate_token_edit_distance(row['llm_response'], row['ground_truth'])
                        
                        doc_f1_scores.append(f1)
                        doc_edit_distances.append(edit_dist)
                        
                        all_f1_scores.append(f1)
                        all_edit_distances.append(edit_dist)
                
                # Store document-level metrics
                document_scores[doc_number] = {
                    'mean_f1': np.mean(doc_f1_scores),
                    'mean_edit_distance': np.mean(doc_edit_distances),
                    'num_questions': len(doc_f1_scores)
                }
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Calculate overall scores
        overall_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
        overall_edit_distance = np.mean(all_edit_distances) if all_edit_distances else 0.0
        
        # Create DataFrame with document-level metrics
        results_data = []
        for doc_num, scores in document_scores.items():
            results_data.append({
                'document_number': doc_num,
                'mean_f1_score': scores['mean_f1'],
                'mean_token_edit_distance': scores['mean_edit_distance']
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Add overall summary row
        summary_row = pd.DataFrame([{
            'document_number': 'OVERALL',
            'mean_f1_score': overall_f1,
            'mean_token_edit_distance': overall_edit_distance
        }])
        
        # Concatenate the results with the summary row
        results_df = pd.concat([results_df, summary_row], ignore_index=True)
        
        # Save to CSV
        results_df.to_csv('dataframe_results/benchmark_results.csv', index=False)
        
        return document_scores, overall_f1, overall_edit_distance

if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    doc_scores, overall_f1, overall_edit_distance = evaluator.evaluate_documents()
    
    print("\nDocument-level scores:")
    for doc_num, scores in doc_scores.items():
        print(f"Document {doc_num}:")
        print(f"  Mean F1: {scores['mean_f1']:.4f}")
        print(f"  Mean Token Edit Distance: {scores['mean_edit_distance']:.4f}")
        print(f"  Questions evaluated: {scores['num_questions']}")
    
    print(f"\nOverall F1 score: {overall_f1:.4f}")
    print(f"Overall Token Edit Distance: {overall_edit_distance:.4f}")
    print(f"\nDetailed results have been saved to: dataframe_results/benchmark_results.csv")
