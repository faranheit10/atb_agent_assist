from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

for m in client.models.list():
    print(f"Model ID: {m.name}, Supported Actions: {m.supported_actions}")
