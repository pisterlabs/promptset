import openai
import os
import regex as re

api_key = os.environ.get("OPENAI_API_KEY")

def get_text_response(prompt):
    openai.api_key = api_key
    
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message.content

def save_to_file(file_name, code):
    with open(file_name, 'w') as file:
        file.write(code)

def generate_scraper(prompt):
    response_text = get_text_response(prompt)
    scraper_code = re.search(r"(?<=```python\n)[\s\S]+?(?=\n```)", response_text).group(0)
    save_to_file("scraper_code.py", scraper_code)
    return scraper_code