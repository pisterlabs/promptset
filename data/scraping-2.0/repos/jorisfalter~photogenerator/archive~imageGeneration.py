from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# OpenAI API Key
api_key = os.getenv('API_KEY')

client = OpenAI(api_key=api_key)

response = client.images.generate(
    model="dall-e-3",
    prompt="This image shows a simple hand-drawn sketch, which appears to represent an airplane from a front view. The drawing includes two wings on either side with what may be interpreted as engines, a fuselage with a cockpit, and landing gear or wheels at the bottom. There are also several vertical lines above the airplane that could signify either rain or motion lines. The sketch is quite basic and lacks significant detail but conveys the essential features of an aircraft.",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
print(image_url)
