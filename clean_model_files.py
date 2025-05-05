import glob
import os
import pandas as pd

from collections import defaultdict

def clean_all_model_outputs(cleaned_dir: str, original_dir: str, output_dir: str):
    cleaned_files = glob.glob(os.path.join(cleaned_dir, "L*_cleaned.csv"))
    cleaned_dict = {}

    for path in cleaned_files:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (row['document_number'], row['question'].strip().lower())
            cleaned_dict[key] = row

    model_files = glob.glob(os.path.join(original_dir, "L*_*.csv"))
    cleaned_outputs = defaultdict(list)
    used_doc_ids = set()

    for path in model_files:
        filename = os.path.basename(path)
        try:
            level_str, model = filename.split("_", 1)
            level = int(level_str[1:])
            model = model.replace(".csv", "")
        except ValueError:
            print(f"Skipping unrecognized file: {filename}")
            continue

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            key = (row['document_number'], row['question'].strip().lower())
            if key in cleaned_dict:
                cleaned = cleaned_dict[key]
                used_doc_ids.add(row['document_number'])
                new_level = int(cleaned['num_hops'] + cleaned['num_set_operations'] + cleaned['multiple_answer_dimension'])
                new_row = row.copy()
                new_row['answer'] = cleaned['answer']
                new_row['num_hops'] = cleaned['num_hops']
                new_row['num_set_operations'] = cleaned['num_set_operations']
                new_row['multiple_answers'] = cleaned['multiple_answer_dimension']
                new_row['sum_values'] = new_level
                cleaned_outputs[(new_level, model)].append(new_row)

    os.makedirs(output_dir, exist_ok=True)
    for (level, model), rows in cleaned_outputs.items():
        df_out = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, f"L{level}_{model}_cleaned.csv")
        df_out.to_csv(out_path, index=False)

    return sorted(used_doc_ids)


def main():
    cleaned_dir = './cleaned'
    original_dir = './original'
    output_dir = './output'
    clean_all_model_outputs(cleaned_dir, original_dir, output_dir)


if __name__ == '__main__':
    main()
