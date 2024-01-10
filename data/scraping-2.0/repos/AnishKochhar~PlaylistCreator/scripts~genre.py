import os
from openai import OpenAI
from dotenv import load_dotenv

'''
Helper script that uses ChatGPT to generate genre combinations for different 
combination counts
'''

load_dotenv()

client = OpenAI (
    api_key=os.environ.get("OPENAI_API_KEY")
)

chat_completion = client.chat.completions.create (
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ]
)
# if __name__ == "__main__":
#     for combination_count in range(2, 6):
#         pass