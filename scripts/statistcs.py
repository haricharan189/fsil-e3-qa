import os
import pandas as pd

def classify_question(row):
    try:
        difficulty_score = (
            int(row.get('num_hops', 0)) +
            int(row.get('num_set_operations', 0)) +
            int(row.get('multiple_answer_dimension', 0))
        )
    except Exception as e:
        print(f"Error calculating difficulty for row: {e}")
        return 'other'

    if difficulty_score == 1:
        return 'easy'
    elif 2 <= difficulty_score <= 4:
        return 'medium'
    elif difficulty_score == 5:
        return 'hard'
    else:
        return 'other'

def process_directory(directory):
    all_stats = []

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath, sep='\t')
                print(f"\nProcessing file: {filename}")
                print("Columns:", df.columns.tolist())

                # Add 'difficulty' column only if required columns are present
                if {'num_hops', 'num_set_operations', 'multiple_answer_dimension'}.issubset(df.columns):
                    df['difficulty'] = df.apply(classify_question, axis=1)
                else:
                    print(f"Skipping difficulty classification for {filename} due to missing columns.")
                    df['difficulty'] = 'other'

                # Add document name
                df['document'] = filename
                all_stats.append(df)

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

# === UPDATE WITH YOUR DEV DIR ===
dev_dir = "/Users/vidhyakshayakannan/Downloads/data/dev"

# === Run processing ===
dev_stats = process_directory(dev_dir)

# === Print summary ===
print("\n=== Summary by Difficulty ===")
print(dev_stats['difficulty'].value_counts())

# === Optionally save the result ===
output_path = os.path.join(dev_dir, "dev_stats_with_difficulty.csv")
dev_stats.to_csv(output_path, sep='\t', index=False)
print(f"\nSaved stats to {output_path}")
