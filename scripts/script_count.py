import pandas as pd
import glob
import os
from collections import defaultdict

# Directory containing one-CSV-per-template
directory = "/Users/vidhyakshayakannan/fsil-e3-qa/template_wise"

# Grab all CSVs (each CSV = one template)
csv_files = glob.glob(os.path.join(directory, "*.csv"))

# Map: document_number -> set of templates it appears in
doc_templates = defaultdict(set)

for file in csv_files:
    # Use filename (sans “.csv”) as your template identifier
    template_name = os.path.splitext(os.path.basename(file))[0]
    try:
        df = pd.read_csv(file)
        # For each unique doc in this template, record that template
        for doc in df['document_number'].dropna().unique():
            doc_templates[doc].add(template_name)
    except Exception as e:
        print(f"⚠️  Error reading {file}: {e}")

# Now find which document_number spans the most templates
if doc_templates:
    # Sort all docs by how many templates they appear in (descending)
    sorted_docs = sorted(doc_templates.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Top document
    top_doc, top_templates = sorted_docs[0]
    print(f"📄 document_number with most templates: {top_doc}")
    print(f"🧩 Number of templates: {len(top_templates)}")
    print(f"🔖 Templates: {sorted(top_templates)}")
    
    # (Optionally) show runner‑up
    if len(sorted_docs) > 1:
        runner_up_doc, runner_up_templates = sorted_docs[1]
        print(f"\n🏅 Runner‑up document_number: {runner_up_doc}")
        print(f"🧩 Templates: {sorted(runner_up_templates)}")
else:
    print("No document_number entries found in any template-CSV.")
