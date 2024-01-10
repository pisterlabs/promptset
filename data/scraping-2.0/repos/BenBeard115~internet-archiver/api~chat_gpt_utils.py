"""Script for Open AI functions."""

from dotenv import load_dotenv
from openai import OpenAI

GPT_3_MODEL = 'gpt-3.5-turbo-1106'
GPT_4_MODEL = 'gpt-4-1106-preview'


load_dotenv()


def read_html_file(file_path: str) -> str:
    """Reads in HTML file."""

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


def generate_summary(html_content, gpt_model: str = GPT_3_MODEL):
    """Generates summary of HTML content"""

    max_html_tokens = 10000 if gpt_model == GPT_3_MODEL else 50000
    html_content = html_content[:max_html_tokens]

    client = OpenAI()

    prompt = """You are an elite web content curator, renowned for crafting compelling and succinct summaries of web pages. Your expertise lies in distilling the essence of a webpage's content and function, with a primary focus on conveying what the page is about. 

Your task is to create a summary of the HTML document you receive, capturing the essence of the webpage's content in a way that is informative and engaging for users of our internet archiver website. 🌐✨ 

Ensure your summary is both captivating and concise, as it will be stored in our database for users to access. Kickstart the description by highlighting the core theme or purpose of the page, enticing users to explore further. Feel free to incorporate emojis to add a touch of vibrancy.

Your mission is to make each summary an invitation, sparking curiosity and encouraging users to delve into the fascinating world captured within each archived webpage. 📚💻
"""

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Please summarise this webpage as instructed: {html_content}."}
        ]
    )

    return completion.choices[0].message.content


if __name__ == "__main__":

    filenames = ['pete_bradshaw', 'rains']
    for filename in filenames:
        html_file = f'static/{filename}.html'

        html_content = read_html_file(html_file)

        print(filename)
        print(generate_summary(html_content))

        print()
