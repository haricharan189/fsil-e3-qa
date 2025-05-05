import os
import pandas as pd
from collections import defaultdict

input_folder = "./output"
output_folder = "./restructured"

# Map model name to lists of files by difficulty
files_by_model = defaultdict(lambda: {"easy": None, "medium": [], "hard": None})

# Parse all files
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        parts = filename.split("_", 2)  # e.g. ["L1", "test", "model.csv"]
        level = int(parts[0][1])
        model_name = parts[2].replace(".csv", "")
        full_path = os.path.join(input_folder, filename)

        if level == 1:
            files_by_model[model_name]["easy"] = full_path
        elif 2 <= level <= 4:
            files_by_model[model_name]["medium"].append(full_path)
        elif level == 5:
            files_by_model[model_name]["hard"] = full_path

# Write renamed and merged files
os.makedirs(output_folder, exist_ok=True)

for model, parts in files_by_model.items():
    if parts["easy"]:
        df = pd.read_csv(parts["easy"])
        df.to_csv(os.path.join(output_folder, f"easy_{model}.csv"), index=False)

    if parts["medium"]:
        dfs = [pd.read_csv(f) for f in parts["medium"]]
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined.to_csv(os.path.join(output_folder, f"medium_{model}.csv"), index=False)

    if parts["hard"]:
        df = pd.read_csv(parts["hard"])
        df.to_csv(os.path.join(output_folder, f"hard_{model}.csv"), index=False)

