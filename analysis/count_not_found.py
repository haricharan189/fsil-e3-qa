import os
import pandas as pd
import re
from prettytable import PrettyTable

directory_path = "/Users/vidhyakshayakannan/Downloads/benchmarking results"
csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".csv")]

levels = ["L1", "L2", "L3", "L4", "L5"]
level_pattern = re.compile(r"(L\d)_test")
not_found_counts = {lvl: 0 for lvl in levels}
zero_f1_counts = {lvl: 0 for lvl in levels}
low_f1_counts = {lvl: 0 for lvl in levels}
total_questions = {lvl: 0 for lvl in levels}

for file in csv_files:
    try:
        df = pd.read_csv(file)
        df["llm_response"] = df["llm_response"].astype(str).str.lower()

        match = level_pattern.search(os.path.basename(file))
        if match:
            level = match.group(1)
            not_found = df["llm_response"].eq("not found").sum()
            zero_f1 = (df["F1_Score"] == 0).sum()
            low_f1 = (df["F1_Score"] < 0.5).sum()
            total = len(df)

            not_found_counts[level] += not_found
            zero_f1_counts[level] += zero_f1
            low_f1_counts[level] += low_f1
            total_questions[level] += total

            print(f"File: {os.path.basename(file)} -> 'Not Found': {not_found}, Zero F1: {zero_f1}, Low F1: {low_f1}, Total: {total}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

def build_table(title, counts_dict):
    table = PrettyTable()
    table.field_names = ["Level", "Count", "Total", "Percentage"]
    total_count = 0
    total_all = 0

    for level in levels:
        count = counts_dict[level]
        total = total_questions[level]
        percent = (count / total) * 100 if total > 0 else 0
        table.add_row([level, count, total, f"{percent:.2f}%"])
        total_count += count
        total_all += total

    overall_percent = (total_count / total_all) * 100 if total_all > 0 else 0
    table.add_row(["TOTAL", total_count, total_all, f"{overall_percent:.2f}%"])
    print(f"{title} by Level:\n")
    print(table)
    print()

build_table("'Not Found'", not_found_counts)
build_table("Zero F1", zero_f1_counts)
build_table("Low F1 (< 0.5)", low_f1_counts)
