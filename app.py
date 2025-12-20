import os
import requests
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage


from agent import agent_app

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# --- CONFIG SWITCHER ---
# This acts as your "Switch Case" for different languages
LANGUAGE_CONFIG = {
    "hindi": {"code": "hi-IN", "speaker": "anushka"},
    "marathi": {"code": "mr-IN", "speaker": "anushka"}, 
    "telugu": {"code": "te-IN", "speaker": "anushka"},
    "odiya": {"code": "od-IN", "speaker": "anushka"},
    "bengali": {"code": "bn-IN", "speaker": "anushka"},
    "tamil": {"code": "ta-IN", "speaker": "anushka"}
}

def transcribe_audio(audio_bytes, lang_key):
    """Sends audio to Sarvam with the specific language hint."""
    url = "https://api.sarvam.ai/speech-to-text"
    
    # Get the code (e.g., 'mr-IN')
    config = LANGUAGE_CONFIG.get(lang_key, LANGUAGE_CONFIG["hindi"])
    
    files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {
        'model': 'saarika:v2.5',
        'language_code': config["code"], # Dynamic Language Switching!
        'with_diarization': 'false'
    }
    headers = {'api-subscription-key': SARVAM_API_KEY}

    try:
        response = requests.post(url, files=files, data=data, headers=headers)
        if response.status_code == 200:
            return response.json().get("transcript", "")
        else:
            print(f"STT Error: {response.text}")
            return None
    except Exception as e:
        print(f"STT Exception: {str(e)}")
        return None

def text_to_speech(text, lang_key):
    """Sends text to Sarvam using the correct speaker and language."""
    url = "https://api.sarvam.ai/text-to-speech"
    
    config = LANGUAGE_CONFIG.get(lang_key, LANGUAGE_CONFIG["hindi"])
    
    payload = {
        "inputs": [text],
        "target_language_code": config["code"], # Dynamic Target
        "speaker": config["speaker"],
        "pitch": 0, "pace": 1.0, "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v2"
    }
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["audios"][0]
    else:
        print(f"TTS Error: {response.text}")
        return None

# --- UPDATED CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(
    file: UploadFile = File(...), 
    language: str = Form(...) 
):
    print(f"🎤 Received Audio in language: {language}")
    
    # 1. STT
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes, language)
    
    if not user_text:
        return JSONResponse(status_code=500, content={"error": "Failed to transcribe"})
    
    print(f"🗣️ User Said: {user_text}")

    # 2. Dynamic Persona (With Gender & Language Rules)
    SYSTEM_PROMPT = f"""
    You are SevaSetu, a helpful female government scheme assistant.
    
    CRITICAL INSTRUCTIONS:
    1. Language: You MUST reply in '{language}' ONLY.
    2. Gender: You are FEMALE. Use female grammar (e.g., say "Main karti hun", NOT "Main karta hun").
    3. Length: Keep answers SHORT (maximum 2 sentences).
    4. Format: Do not use markdown (like **bold**). Just plain text.
    """
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text)
    ]
    
    final_response = ""
    
    # --- LOGIC UPDATE: CLEAN THE RESPONSE ---
    try:
        for event in agent_app.stream({"messages": messages}):
            for key, value in event.items():
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    
                    if hasattr(last_msg, "content") and last_msg.content:
                        raw_content = last_msg.content
                        
                        # CASE A: It's a simple string (Normal)
                        if isinstance(raw_content, str):
                            final_response = raw_content
                            
                        # CASE B: It's a List (The error you saw)
                        elif isinstance(raw_content, list):
                            text_parts = []
                            for item in raw_content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    text_parts.append(item.get('text'))
                            final_response = " ".join(text_parts)
                            
    except Exception as e:
        print(f"Agent Error: {e}")
        final_response = "Technical error. Please try again."

    # Fallback if empty
    if not final_response or final_response.strip() == "":
        print("⚠️ Warning: Agent returned empty text.")
        final_response = "Sorry, I could not understand. Please speak again."

    print(f"🤖 Agent Replied (Cleaned): {final_response}")

    # 3. TTS
    audio_b64 = text_to_speech(final_response, language)
    
    return {
        "user_text": user_text,
        "agent_text": final_response,
        "audio": audio_b64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)