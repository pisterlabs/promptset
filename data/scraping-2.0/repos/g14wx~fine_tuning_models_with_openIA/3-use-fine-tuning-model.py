from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)
current_model_id = os.getenv('FINE_TUNING_MODEL_ID')
system_content = os.getenv("SYSTEM_ROLE_CONTENT")
user_content = os.getenv("USER_ROLE_CONTENT")

response = client.chat.completions.create(
    model=current_model_id,
    messages=[
        {
            "role": "system",
            "content": system_content
        },
        {"role": "user", "content": user_content}
    ],
)
print(response.choices[0].message)
