import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY missing!")

#Setup Embeddings 
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "scheme-index"

#Load Data 
print("Loading schemes.json...")
with open("schemes.json", "r") as f:
    data = json.load(f)

documents = []
for item in data:
    page_content = f"Scheme: {item['name']}. Description: {item['description']} Eligibility: {item['eligibility']} Documents: {item['documents']}"
    
    metadata = {"name": item["name"], "documents": item["documents"]}
    
    documents.append(Document(page_content=page_content, metadata=metadata))

print(f"Uploading {len(documents)} schemes to Pinecone...")
vector_store = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

print("Success! Database populated.")