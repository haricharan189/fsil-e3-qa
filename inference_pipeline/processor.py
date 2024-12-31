"""
Changes:
  - Writes output CSV to dataframe/ground_truth_qa.csv
  - Reads ground_truth files from ground_truth/.
"""

import pandas as pd
import os
import glob
import re

import config

class GroundTruthProcessor:
    def __init__(self, ground_truth_dir=None):
        self.ground_truth_dir = ground_truth_dir or config.GROUND_TRUTH_DIR
        self.dataframe_dir    = config.DATAFRAME_DIR
        os.makedirs(self.dataframe_dir, exist_ok=True)

    def extract_document_number(self, filename):
        base_name = os.path.basename(filename)
        match = re.search(r'answers_(\d+)\.txt$', base_name)
        if match:
            return match.group(1)
        return None

    def process_ground_truth_files(self):
        data = []
        
        l1_files = glob.glob(os.path.join(self.ground_truth_dir, "L1_answers_*.txt"))
        l2_files = glob.glob(os.path.join(self.ground_truth_dir, "L2_answers_*.txt"))
        
        for file_path in l1_files:
            doc_num = self.extract_document_number(file_path)
            if doc_num:
                data.extend(self.process_file(file_path, doc_num, level=1))
            
        for file_path in l2_files:
            doc_num = self.extract_document_number(file_path)
            if doc_num:
                data.extend(self.process_file(file_path, doc_num, level=2))
            
        df = pd.DataFrame(data, columns=['document_number', 'question', 'answer', 'level'])
        df['document_number'] = df['document_number'].astype(int)
        df = df.sort_values('document_number')
        
        output_path = os.path.join(self.dataframe_dir, "ground_truth_qa.csv")
        df.to_csv(output_path, index=False)
        
        return df

    def process_file(self, file_path, doc_num, level):
        data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i in range(0, len(lines)-1, 2):
            question = lines[i].strip()
            answer   = lines[i+1].strip()
            
            if question.startswith('Q:'):
                question = question[2:].strip()
                if answer.startswith('A:'):
                    answer = answer[2:].strip()
                data.append((doc_num, question, answer, level))
                
        return data

def main():
    processor = GroundTruthProcessor()
    df = processor.process_ground_truth_files()
    print(f"Created DataFrame with {len(df)} rows.")
    print(df.head())

if __name__ == "__main__":
    main()
