import os
import re
import requests
import openai
from bs4 import BeautifulSoup
from collections import defaultdict

# Read the OpenAI API key from the environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Rest of the script remains the same


def load_bookmarks(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def extract_links(bookmarks_html):
    soup = BeautifulSoup(bookmarks_html, "html.parser")
    links = soup.find_all("a")
    return links

def check_alive(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def get_summary(url):
    prompt = f"Please provide a short summary of the content at this URL: {url}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    summary = response.choices[0].text.strip()
    return summary

def categorize_bookmarks(links):
    categories = defaultdict(list)
    for link in links:
        url = link["href"]
        title = link.get_text()

        if not check_alive(url):
            continue

        summary = get_summary(url)
        prompt = f"Given this summary, '{summary}', what category would you assign to the website?"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.7,
        )
        category = response.choices[0].text.strip()
        categories[category].append((title, url, summary))

    return categories

def generate_html_output(categories, user_characteristics):
    bootstrap_cdn = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analyzed Bookmarks</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    </head>
    <body>
    <div class="container">
    '''

    content = f'<h1 class="text-center my-3">Analyzed Bookmarks</h1>'
    content += f'<h3 class="text-center my-3">User Characteristics</h3>'
    content += f'<p>{user_characteristics}</p>'
    content += '<h3 class="text-center my-3">Bookmarks by Category</h3>'

    for category, bookmarks in categories.items():
        content += f'<h4>{category}</h4>'
        content += '<ul>'
        for title, url, summary in bookmarks:
            content += f'<li><a href="{url}" target="_blank">{title}</a> - {summary}</li>'
        content += '</ul>'

    content += '</div></body></html>'

    with open("analyzed_bookmarks.html", "w", encoding="utf-8") as output_file:
        output_file.write(bootstrap_cdn + content)

def get_user_characteristics(bookmarks):
    prompt = "Based on this collection of bookmarks, describe the characteristics of the person who collected them:\n"
    for title, url in bookmarks:
        prompt += f"- {title}: {url}\n"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    characteristics = response.choices[0].text.strip()
    return characteristics

def main():
    bookmarks_html = load_bookmarks("bookmarks_personal.html")
    links = extract_links(bookmarks_html)
    bookmarks = [(link.get_text(), link["href"]) for link in links if check_alive(link["href"])]
    categories = categorize_bookmarks(links)
    user_characteristics = get_user_characteristics(bookmarks)
    generate_html_output(categories, user_characteristics)

if __name__ == "__main__":
    main()
