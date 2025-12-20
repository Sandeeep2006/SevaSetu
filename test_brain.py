import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

google_key = os.getenv("GEMINI_API_KEY")
sarvam_key = os.getenv("SARVAM_API_KEY")

if not google_key:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY is missing from .env file!")
if not sarvam_key:
    raise ValueError("CRITICAL ERROR: SARVAM_API_KEY is missing from .env file!")

print("Keys found. Proceeding...")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


native_language = "Marathi" 
query = "I am a 60-year-old farmer with low income. Tell me in 2 sentences if I might get a pension."

prompt = f"Answer the following query in {native_language} language only: {query}"

try:
    response = llm.invoke(prompt)
    print(f"--- {native_language} Response ---")
    print(response.content)
except Exception as e:
    print(f"Error: {e}. Check if your API key is correct!")