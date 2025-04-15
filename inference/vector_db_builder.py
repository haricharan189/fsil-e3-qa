import json
import os

import config

from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from main import clean_html


def build_vector_store():
    json_path = os.path.join(config.JSON_PATH, config.JSON_FILE)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        doc_id = str(entry.get("id"))
        html = entry.get("data", {}).get("html", "")
        text = clean_html(html)
        chunks = [text[i:i + config.RAG_CHUNK_SIZE]
                  for i in range(0, len(text), config.RAG_CHUNK_SIZE)]
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={
                        "doc_id": doc_id, "chunk_id": i}))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding=embeddings)
    db.save_local(config.VECTOR_DB_DIR)


if __name__ == "__main__":
    build_vector_store()
