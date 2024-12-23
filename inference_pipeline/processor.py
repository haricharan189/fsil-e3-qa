import pandas as pd
import os
import glob
import re

class GroundTruthProcessor:
    def __init__(self, ground_truth_dir="ground_truth"):
        """
        Initialize the processor with the directory containing ground truth files.
        
        Args:
            ground_truth_dir (str): Path to the directory containing ground truth files
        """
        self.ground_truth_dir = ground_truth_dir
        self.dataframe_dir = "dataframe"
        os.makedirs(self.dataframe_dir, exist_ok=True)

    def extract_document_number(self, filename):
        """
        Extract document number from filename (e.g., 'L1_answers_216.txt' -> '216').
        """
        # Extract the base filename
        base_name = os.path.basename(filename)
        # Look for pattern 'answers_XXX.txt'
        match = re.search(r'answers_(\d+)\.txt$', base_name)
        if match:
            return match.group(1)
        return None

    def process_ground_truth_files(self):
        """Process all ground truth files and create a DataFrame."""
        data = []
        
        # Get all ground truth files
        l1_files = glob.glob(os.path.join(self.ground_truth_dir, "L1_answers_*.txt"))
        l2_files = glob.glob(os.path.join(self.ground_truth_dir, "L2_answers_*.txt"))
        
        # Process L1 files
        for file_path in l1_files:
            doc_num = self.extract_document_number(file_path)
            if doc_num:  # Only process if we have a valid document number
                data.extend(self.process_file(file_path, doc_num, level=1))
            
        # Process L2 files
        for file_path in l2_files:
            doc_num = self.extract_document_number(file_path)
            if doc_num:  # Only process if we have a valid document number
                data.extend(self.process_file(file_path, doc_num, level=2))
            
        # Create DataFrame
        df = pd.DataFrame(data, columns=['document_number', 'question', 'answer', 'level'])
        
        # Convert document_number to integer
        df['document_number'] = df['document_number'].astype(int)
        
        # Sort by document number
        df = df.sort_values('document_number')
        
        # Save DataFrame
        output_path = os.path.join(self.dataframe_dir, "ground_truth_qa.csv")
        df.to_csv(output_path, index=False)
        
        return df

    def process_file(self, file_path, doc_num, level):
        """
        Process individual ground truth file and extract Q&A pairs.
        
        Args:
            file_path (str): Path to the ground truth file
            doc_num (str): Document number
            level (int): Level of the questions (1 or 2)
            
        Returns:
            list: List of tuples containing (document_number, question, answer, level)
        """
        data = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Process lines in pairs (question followed by answer)
        for i in range(0, len(lines)-1, 2):
            question = lines[i].strip()
            answer = lines[i+1].strip()
            
            # Skip empty lines and ensure the line starts with Q: for questions
            if question.startswith('Q:'):
                # Remove 'Q: ' prefix from question
                question = question[2:].strip()
                # Remove 'A: ' prefix from answer if present
                answer = answer[2:].strip() if answer.startswith('A:') else answer.strip()
                
                data.append((doc_num, question, answer, level))
                
        return data

if __name__ == "__main__":
    processor = GroundTruthProcessor()
    df = processor.process_ground_truth_files()
    print(f"Created DataFrame with {len(df)} rows")
    print("\nSample of the DataFrame:")
    print(df.head())
