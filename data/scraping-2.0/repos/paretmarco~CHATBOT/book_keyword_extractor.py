import openai
import json
import time
from config import OPENAI_API_KEY

def load_draft_text():
    try:
        with open("drafts/draft_text.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("draft_text.txt not found.")
        return ""

openai.api_key = OPENAI_API_KEY
MODEL = "gpt-4"

def generate_keywords(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=prompt,
                temperature=0.7,
                max_tokens=200,
                stop=None,
            )
            keywords = response.choices[0].message.content.strip()
            return keywords
        except Exception as e:
            print(f"Error during API call: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this keyword extraction.")
                return "Error: Unable to generate keywords."

text = load_draft_text()

keyword_prompt = [
    {"role": "system", "content": "You are an AI language model and you can extract the most important keywords from a given text."},
    {"role": "user", "content": f"Extract the most important keywords from the following text: '{text}' and write them with a comma (,) between each one"}
]

keywords = generate_keywords(keyword_prompt)
print("Extracted keywords:", keywords)

# Save the extracted keywords to the file drafts/keywords.txt
with open("drafts/keywords.txt", "w", encoding="utf-8") as f:
    f.write(keywords)
    print("Keywords saved to drafts/keywords.txt.")
