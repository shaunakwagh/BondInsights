from langchain.vectorstores import FAISS
import os
def get_retriever(texts, embeddings):
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    return retriever