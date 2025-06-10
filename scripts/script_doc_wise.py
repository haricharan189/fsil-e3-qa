import csv
from collections import defaultdict
import os

# Load the CSV file
csv_file = 'level1.csv'

# Dictionary to store list of (question, answer) tuples by document_number
questions_by_doc = defaultdict(list)

# Read and group questions & answers
with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        doc_num = row['document_number']
        question = row['question']
        answer = row['answer']
        questions_by_doc[doc_num].append((question, answer))

# Create output directory
output_dir = 'document_questions'
os.makedirs(output_dir, exist_ok=True)

# Write each document's Q&A to a separate file
for doc_num, qa_list in questions_by_doc.items():
    filename = os.path.join(output_dir, f'document_{doc_num}.txt')
    with open(filename, 'w', encoding='utf-8') as f:
        for q, a in qa_list:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n")
            f.write("\n")  # blank line between entries

print("Questions saved document-wise successfully.")
