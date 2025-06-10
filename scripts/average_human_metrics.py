import pandas as pd
import os
from glob import glob

# === Load original CSV ===
main_file_path = '/Users/vidhyakshayakannan/Documents/sample_vidhya_rating_with_all_judge.csv'
main_df = pd.read_csv(main_file_path)
main_df['question'] = main_df['question'].astype(str).str.strip().str.lower()

# Add column if not already there
if 'avg_llm_human_rating' not in main_df.columns:
    main_df['avg_llm_human_rating'] = None

# === Load human rating files ===
human_rating_dir = '/Users/vidhyakshayakannan/Documents/human_ratings'
human_files = glob(os.path.join(human_rating_dir, '*.csv'))[:3]  # Only first 3 files

# Collect all ratings
ratings = []

for file in human_files:
    try:
        df = pd.read_csv(file)
        if 'question' not in df.columns or 'llm_human_rating' not in df.columns:
            print(f"⚠️ Skipping file (missing columns): {file}")
            continue
        df['question'] = df['question'].astype(str).str.strip().str.lower()
        ratings.append(df[['question', 'llm_human_rating']])
        print(f"✅ Loaded: {os.path.basename(file)}")
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# === Combine and compute average rating per question ===
if ratings:
    combined = pd.concat(ratings)
    avg_ratings = combined.groupby('question', as_index=False)['llm_human_rating'].mean()
    avg_ratings.rename(columns={'llm_human_rating': 'avg_llm_human_rating'}, inplace=True)

    # === Merge with main_df ===
    merged = pd.merge(main_df, avg_ratings, on='question', how='left', suffixes=('', '_new'))
    main_df['avg_llm_human_rating'] = main_df['avg_llm_human_rating'].combine_first(merged['avg_llm_human_rating_new'])

    # === Save to file ===
    output_file_path = '/Users/vidhyakshayakannan/Documents/sample_vidhya_rating_with_human_avg.csv'
    main_df.to_csv(output_file_path, index=False)
    print(f"\n✅ Updated file saved to: {output_file_path}")
else:
    print("❌ No valid human rating files found.")
