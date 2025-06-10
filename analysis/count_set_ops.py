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
        required_cols = {"num_hops", "num_set_operations", "llm_response", "F1_Score", "sum_values", "Edit_Distance_Sim", "Cosine_Similarity"}
        if required_cols.issubset(df.columns):
            df["llm_response"] = df["llm_response"].astype(str).str.lower()

            for hop_val in hops_stats:
                hop_df = df[df["num_hops"] == hop_val]
                stats = hops_stats[hop_val]
                stats["total"] += len(hop_df)
                stats["not_found"] += (hop_df["llm_response"] == "not found").sum()
                stats["low_f1"] += (hop_df["F1_Score"] < 0.5).sum()
                stats["low_cosine"] += (hop_df["Cosine_Similarity"] < LOW_COSINE_THRESHOLD).sum()
                stats["high_edit"] += (hop_df["Edit_Distance_Sim"] > HIGH_EDIT_THRESHOLD).sum()

            for ops_val in set_ops_stats:
                ops_df = df[df["num_set_operations"] == ops_val]
                stats = set_ops_stats[ops_val]
                stats["total"] += len(ops_df)
                stats["not_found"] += (ops_df["llm_response"] == "not found").sum()
                stats["low_f1"] += (ops_df["F1_Score"] < 0.5).sum()
                stats["low_cosine"] += (ops_df["Cosine_Similarity"] < LOW_COSINE_THRESHOLD).sum()
                stats["high_edit"] += (ops_df["Edit_Distance_Sim"] > HIGH_EDIT_THRESHOLD).sum()
    except Exception as e:
        print(f"Error processing {file}: {e}")

from prettytable import PrettyTable

def percent_str(count, total):
    return f"{(count / total * 100):.2f}%" if total > 0 else "0.00%"

# 'Not Found' by Num Hops
hops_nf_table = PrettyTable()
hops_nf_table.field_names = ["Num Hops", "Total", "'Not Found' Count", "'Not Found' %"]
for hop_val in sorted(hops_stats.keys()):
    stats = hops_stats[hop_val]
    hops_nf_table.add_row([hop_val, stats["total"], stats["not_found"], percent_str(stats["not_found"], stats["total"])])

# Low F1 by Num Hops
hops_f1_table = PrettyTable()
hops_f1_table.field_names = ["Num Hops", "Total", "Low F1 (<0.5)", "Low F1 %"]
for hop_val in sorted(hops_stats.keys()):
    stats = hops_stats[hop_val]
    hops_f1_table.add_row([hop_val, stats["total"], stats["low_f1"], percent_str(stats["low_f1"], stats["total"])])

# Low Cosine Sim by Num Hops
hops_cosine_table = PrettyTable()
hops_cosine_table.field_names = ["Num Hops", "Total", "Low Cosine Sim (<0.5)", "Low Cosine %"]
for hop_val in sorted(hops_stats.keys()):
    stats = hops_stats[hop_val]
    hops_cosine_table.add_row([hop_val, stats["total"], stats["low_cosine"], percent_str(stats["low_cosine"], stats["total"])])

# High Edit Distance by Num Hops
hops_edit_table = PrettyTable()
hops_edit_table.field_names = ["Num Hops", "Total", "High Edit Dist (>0.5)", "High Edit %"]
for hop_val in sorted(hops_stats.keys()):
    stats = hops_stats[hop_val]
    hops_edit_table.add_row([hop_val, stats["total"], stats["high_edit"], percent_str(stats["high_edit"], stats["total"])])

print("'Not Found' by Num Hops:")
print(hops_nf_table)
print("\nLow F1 by Num Hops:")
print(hops_f1_table)
print("\nLow Cosine Similarity by Num Hops:")
print(hops_cosine_table)
print("\nHigh Edit Distance by Num Hops:")
print(hops_edit_table)

# 'Not Found' by Set Operations
setops_nf_table = PrettyTable()
setops_nf_table.field_names = ["Set Ops", "Total", "'Not Found' Count", "'Not Found' %"]
for ops_val in sorted(set_ops_stats.keys()):
    stats = set_ops_stats[ops_val]
    setops_nf_table.add_row([ops_val, stats["total"], stats["not_found"], percent_str(stats["not_found"], stats["total"])])

# Low F1 by Set Operations
setops_f1_table = PrettyTable()
setops_f1_table.field_names = ["Set Ops", "Total", "Low F1 (<0.5)", "Low F1 %"]
for ops_val in sorted(set_ops_stats.keys()):
    stats = set_ops_stats[ops_val]
    setops_f1_table.add_row([ops_val, stats["total"], stats["low_f1"], percent_str(stats["low_f1"], stats["total"])])

# Low Cosine Sim by Set Operations
setops_cosine_table = PrettyTable()
setops_cosine_table.field_names = ["Set Ops", "Total", "Low Cosine Sim (<0.5)", "Low Cosine %"]
for ops_val in sorted(set_ops_stats.keys()):
    stats = set_ops_stats[ops_val]
    setops_cosine_table.add_row([ops_val, stats["total"], stats["low_cosine"], percent_str(stats["low_cosine"], stats["total"])])

# High Edit Distance by Set Operations
setops_edit_table = PrettyTable()
setops_edit_table.field_names = ["Set Ops", "Total", "High Edit Dist (>0.5)", "High Edit %"]
for ops_val in sorted(set_ops_stats.keys()):
    stats = set_ops_stats[ops_val]
    setops_edit_table.add_row([ops_val, stats["total"], stats["high_edit"], percent_str(stats["high_edit"], stats["total"])])

print("'Not Found' by Set Operations:")
print(setops_nf_table)
print("\nLow F1 by Set Operations:")
print(setops_f1_table)
print("\nLow Cosine Similarity by Set Operations:")
print(setops_cosine_table)
print("\nHigh Edit Distance by Set Operations:")
print(setops_edit_table)

def plot_combined_metrics(x_labels, stats_dict, x_label_name, filename=None):
    x = np.arange(len(x_labels))

    not_found = np.array([
        stats_dict[k]["not_found"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])
    low_f1 = np.array([
        stats_dict[k]["low_f1"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])
    low_cosine = np.array([
        stats_dict[k]["low_cosine"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])
    high_edit = np.array([
        stats_dict[k]["high_edit"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])

    plt.figure(figsize=(10, 6))

    plt.plot(x, not_found, marker='o', color='#8da0cb', label="Not Found (%)", linewidth=2)
    plt.fill_between(x, not_found - 2, not_found + 2, color='#8da0cb', alpha=0.2)

    plt.plot(x, low_f1, marker='s', color='#fc8d62', label="Low F1 (<0.5) (%)", linewidth=2)
    plt.fill_between(x, low_f1 - 2, low_f1 + 2, color='#fc8d62', alpha=0.2)

    plt.plot(x, low_cosine, marker='^', color='#a6d854', label="Low Cosine Similarity (<0.5) (%)", linewidth=2)
    plt.fill_between(x, low_cosine - 2, low_cosine + 2, color='#a6d854', alpha=0.2)

    plt.plot(x, high_edit, marker='D', color='#e78ac3', label="High Edit Distance (>0.5) (%)", linewidth=2)
    plt.fill_between(x, high_edit - 2, high_edit + 2, color='#e78ac3', alpha=0.2)

    plt.xticks(x, x_labels, rotation=0)
    plt.xlabel(x_label_name, fontsize=12, fontweight='bold')
    plt.ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    plt.title("Error Metrics Across Number of Set Operations", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format='pdf')
    plt.show()


def plot_combined_metrics_num_hops(x_labels, stats_dict, x_label_name, filename=None):
    x = np.arange(len(x_labels))

    not_found = np.array([
        stats_dict[k]["not_found"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])
    low_f1 = np.array([
        stats_dict[k]["low_f1"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])
    low_cosine = np.array([
        stats_dict[k]["low_cosine"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])
    high_edit = np.array([
        stats_dict[k]["high_edit"] / stats_dict[k]["total"] * 100 if stats_dict[k]["total"] > 0 else 0
        for k in x_labels
    ])

    plt.figure(figsize=(10, 6))

    padding = 1

    plt.plot(x, not_found, marker='o', color='#8da0cb', label="Not Found (%)", linewidth=2)
    plt.fill_between(x, not_found - padding, not_found + padding, color='#8da0cb', alpha=0.2)

    plt.plot(x, low_f1, marker='s', color='#fc8d62', label="Low F1 (<0.5) (%)", linewidth=2)
    plt.fill_between(x, low_f1 - padding, low_f1 + padding, color='#fc8d62', alpha=0.2)

    plt.plot(x, low_cosine, marker='^', color='#a6d854', label="Low Cosine Similarity (<0.5) (%)", linewidth=2)
    plt.fill_between(x, low_cosine - padding, low_cosine + padding, color='#a6d854', alpha=0.2)

    plt.plot(x, high_edit, marker='D', color='#e78ac3', label="High Edit Distance (>0.5) (%)", linewidth=2)
    plt.fill_between(x, high_edit - padding, high_edit + padding, color='#e78ac3', alpha=0.2)

    plt.xticks(x, x_labels, rotation=0)
    plt.xlabel(x_label_name, fontsize=12, fontweight='bold')
    plt.ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    plt.title("Error Metrics Across Number of Hops", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format='pdf')
    plt.show()


# Call functions with filenames to save PDFs
x_labels_setops = sorted(set_ops_stats.keys())
plot_combined_metrics(x_labels_setops, set_ops_stats, "Set Operations", filename="error_metrics_set_ops.pdf")

x_labels_hops = sorted(hops_stats.keys())
plot_combined_metrics_num_hops(x_labels_hops, hops_stats, "Number of Hops", filename="error_metrics_num_hops.pdf")


