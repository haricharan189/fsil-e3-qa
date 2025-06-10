import pandas as pd
from scipy.stats import kendalltau, pearsonr
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# === Load your data ===
file_path = '/Users/vidhyakshayakannan/Documents/merged_output.csv'
df = pd.read_csv(file_path)

# === Select only relevant numeric columns ===
columns = ['F1_Score', 'Cosine_Similarity', 'Edit_Distance_Sim', 'llm_as_a_judge', 'avg_llm_human_rating']
df = df[columns].dropna()


# === Initialize correlation matrices ===
kendall_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
pearson_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

# === Compute correlations for each pair ===
for col1, col2 in itertools.combinations_with_replacement(columns, 2):
    kendall_corr, _ = kendalltau(df[col1], df[col2])
    pearson_corr, _ = pearsonr(df[col1], df[col2])

    kendall_matrix.loc[col1, col2] = kendall_corr
    kendall_matrix.loc[col2, col1] = kendall_corr

    pearson_matrix.loc[col1, col2] = pearson_corr
    pearson_matrix.loc[col2, col1] = pearson_corr

# === Print Results ===
print("Kendall's Tau Correlation Matrix:")
print(kendall_matrix.round(3))

print("\nPearson Correlation Matrix:")
print(pearson_matrix.round(3))

# Define a custom light pastel palette (e.g., mint green or soft lilac)
pastel_cmap = sns.light_palette("peachpuff", as_cmap=True)  # Try "skyblue", "orchid", "peachpuff" too!

def plot_heatmap(matrix, title, filename=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix.astype(float).round(2),
        annot=True,
        cmap=pastel_cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        square=True
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight') 
        print(f"Saved heatmap as {filename}")
        
    plt.show()
    
kendall_matrix.columns = kendall_matrix.columns.str.lower()
kendall_matrix.index = kendall_matrix.index.str.lower()

pearson_matrix.columns = pearson_matrix.columns.str.lower()
pearson_matrix.index = pearson_matrix.index.str.lower()

plot_heatmap(kendall_matrix, "Kendall's Tau Correlation", "kendall_tau_correlation.pdf")
plot_heatmap(pearson_matrix, "Pearson Correlation", "pearson_correlation.pdf")

