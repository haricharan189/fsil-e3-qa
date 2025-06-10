print("\n--- Detailed Failures for High Set Operations ---\n")
import os
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np


LOW_COSINE_THRESHOLD = 0.5    # cosine sim less than 0.5 is considered low
HIGH_EDIT_THRESHOLD = 0.5     # edit distance greater than 0.5 is considered high


directory_path = "/Users/vidhyakshayakannan/Downloads/cleaned_data"
csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".csv")]
hops_stats = {i: {"total": 0, "not_found": 0, "low_f1": 0, "low_cosine": 0, "high_edit": 0} for i in range(1, 4)}
set_ops_stats = {i: {"total": 0, "not_found": 0, "low_f1": 0, "low_cosine": 0, "high_edit": 0} for i in range(0, 4)}

for file in csv_files:
    try:
        df = pd.read_csv(file)
        if "num_set_operations" in df.columns and "llm_response" in df.columns:
            df["llm_response"] = df["llm_response"].astype(str).str.lower()
            filtered_df = df[
                (df["num_set_operations"] >= 2) &
                (df["llm_response"] != "not found") &
                (
                    (df["F1_Score"] < 0.5) |
                    (df["Cosine_Similarity"] < LOW_COSINE_THRESHOLD) |
                    (df["Edit_Distance_Sim"] > HIGH_EDIT_THRESHOLD)
                )
            ]

            if not filtered_df.empty:
                print(f"\nFailures in {os.path.basename(file)}:")
                for _, row in filtered_df.iterrows():
                    print(f"Q: {row.get('question', 'N/A')}")
                    print(f"Document: {row['document_number']}")
                    print(f"LLM Response: {row['llm_response']}")
                    print(f"Ground Truth: {row['answer']}")
                    print(f"F1: {row['F1_Score']:.2f}, Cosine: {row['Cosine_Similarity']:.2f}, Edit Dist Sim: {row['Edit_Distance_Sim']:.2f}")
                    print("-" * 80)
    except Exception as e:
        print(f"Error processing {file}: {e}")
