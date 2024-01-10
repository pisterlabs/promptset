import time
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def generate_pv(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"give me a brief summary of the meeting in bullet points, this is the meeting content:\n {prompt}",
                },
            ],
        )
    except openai.RateLimitError as e:
        if "Too Many Requests" in str(e):
            # Retry after a delay
            time.sleep(5)
            return generate_pv(prompt)
        else:
            # Handle other errors
            print(f"Error: {e}")
            return None

    print(completion.choices[0].message)
    return completion.choices[0].message
