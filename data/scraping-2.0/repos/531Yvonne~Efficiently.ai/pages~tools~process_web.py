import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
load_dotenv()


def fetch_webcontent(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = ' '.join([p.get_text() for p in soup.find_all('p')])
        return content
    except Exception as e:
        return f"Error fetching content: {str(e)}"


def analyze_content(content):
    prompt = f"Summarize the content: {content}"
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=[{"role": "user",
                                                         "content": f"{prompt}"}])
    return response.choices[0].message.content
