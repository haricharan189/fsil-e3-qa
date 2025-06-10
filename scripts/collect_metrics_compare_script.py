import pandas as pd
import os
from glob import glob

# Load the main CSV
main_df = pd.read_csv('/Users/vidhyakshayakannan/Documents/sample_vidhya_rating.csv')  
main_df['question'] = main_df['question'].str.strip().str.lower()

# Initialize new metric columns if not already present
for col in ['F1_Score', 'Cosine_Similarity', 'Edit_Distance_Sim']:
    if col not in main_df.columns:
        main_df[col] = None

# Directory containing cleaned_data CSVs
cleaned_data_dir = '/Users/vidhyakshayakannan/Downloads/cleaned_data'
csv_files = glob(os.path.join(cleaned_data_dir, '*.csv'))

for file in csv_files:
    cleaned_df = pd.read_csv(file)
    
    # Standardize 'question' column
    if 'question' not in cleaned_df.columns:
        print(f"Skipped file {file}: no 'question' column found.")
        continue

    cleaned_df['question'] = cleaned_df['question'].astype(str).str.strip().str.lower()

    # Check for metric columns and fill missing ones with None
    for col in ['F1_Score', 'Cosine_Similarity', 'Edit_Distance_Sim']:
        if col not in cleaned_df.columns:
            cleaned_df[col] = None

    # Drop duplicates to avoid confusion on merge
    cleaned_df = cleaned_df.drop_duplicates(subset=['question'])

    # Merge and update only missing entries
    merged = pd.merge(main_df, cleaned_df[['question', 'F1_Score', 'Cosine_Similarity', 'Edit_Distance_Sim']],
                      on='question', how='left', suffixes=('', '_new'))

    for col in ['F1_Score', 'Cosine_Similarity', 'Edit_Distance_Sim']:
        main_df[col] = main_df[col].combine_first(merged[f'{col}_new'])

# Save the updated CSV
output_path = '/Users/vidhyakshayakannan/Documents/merged_output.csv'
main_df.to_csv(output_path, index=False)
print(f"Updated file saved to {output_path}")
