import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Initialize Connection to DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = PineconeVectorStore(index_name="scheme-index", embedding=embeddings)

@tool
def check_eligibility(user_details: str):
    """
    Checks eligibility by searching the government scheme database.
    Args:
        user_details: A string summary of the user (e.g., "Farmer, income 20k, age 40").
    """
    # Search the DB for schemes matching these details
    results = vector_store.similarity_search(user_details, k=3)
    
    if not results:
        return "No specific schemes found for this profile."
    
    # Format the output nicely
    response = "Based on your details, here are the top schemes found:\n"
    for doc in results:
        response += f"- {doc.page_content}\n"
        
    return response

@tool
def get_scheme_documents(scheme_name: str):
    """
    Finds the documents required for a specific scheme.
    Args:
        scheme_name: Name of the scheme (e.g., "PM Kisan").
    """
    results = vector_store.similarity_search(scheme_name, k=1)
    if results:
        return f"Details found: {results[0].page_content}"
    return "Scheme details not found."