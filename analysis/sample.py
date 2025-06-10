import os
import pandas as pd

# Directory containing CSV files
directory_path = "/Users/vidhyakshayakannan/fsil-e3-qa/analysis/soted"

# List all CSV files in the directory
csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".csv")]

# List to store filtered data
f1_below_threshold = []

# Loop through each CSV file
for file in csv_files:
    try:
        df = pd.read_csv(file)

        # Ensure 'F1_Score' column exists and filter rows where F1 < 0.5
        if "F1_Score" in df.columns:
            filtered_df = df[df["F1_Score"] < 0.5]
            f1_below_threshold.append(filtered_df)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine all filtered data
if f1_below_threshold:
    combined_df = pd.concat(f1_below_threshold, ignore_index=True)

    # Sample 100 questions (or all if less than 100)
    sampled_df = combined_df.sample(n=min(100, len(combined_df)), random_state=42)

    # Save to CSV
    output_file = "sampled_F1_below_0.5.csv"
    sampled_df.to_csv(output_file, index=False)
    
    print(f"Sampled {len(sampled_df)} questions saved to {output_file}")

else:
    print("No questions with F1_Score < 0.5 found.")
