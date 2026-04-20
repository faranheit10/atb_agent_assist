from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview-customtools",
        contents="Say hello"
    )
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
