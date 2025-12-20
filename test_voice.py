import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

def text_to_speech(text, filename="output.wav"):
    url = "https://api.sarvam.ai/text-to-speech"
    
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN", # Hindi
        "speaker": "anushka",
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v2"
    }
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    print(f"Generating audio for: '{text}'...")
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        # Decode the base64 audio and save to file
        audio_data = base64.b64decode(response.json()["audios"][0])
        with open(filename, "wb") as f:
            f.write(audio_data)
        print(f"✅ Success! Audio saved to '{filename}'. Play it to hear the voice.")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Testing sentence
    test_text = "Namaste! Atal Pension Yojana mein aapko guaranteed monthly pension milti hai."
    text_to_speech(test_text)