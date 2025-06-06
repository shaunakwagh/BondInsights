from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI

def get_qa_chain(retriever):
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def query_llm(query, qa_chain):
    result = qa_chain({"query": query})
    answer = result["result"] if "result" in result else str(result)
    citations = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            page = doc.metadata.get("page", "N/A")
            excerpt = doc.page_content[:400].replace('\n', ' ')
            citations.append(f"Page {page}: {excerpt}...")
    return answer, citations