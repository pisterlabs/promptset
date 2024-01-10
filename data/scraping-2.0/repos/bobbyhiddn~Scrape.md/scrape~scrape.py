import requests
import sys
import os
import re
from bs4 import BeautifulSoup
import click
import openai  # Make sure to install the OpenAI package

# Function to send a message to the OpenAI chatbot model and return its response
def send_message(message_log):
    print("Waiting for API...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message_log,
        max_tokens=2000,
        stop=None,
        temperature=0.7,
    )
    print("Received response from API.")

    for choice in response.choices:
        if "text" in choice:
            return choice.text
    return response.choices[0].message.content

# Fetch the content from the given URL
def fetch_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

# Parse the fetched HTML content
def parse_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string if soup.title else "Untitled"
    body = soup.body.get_text() if soup.body else "No content available."
    return title, body

# Clean the content
def clean_content(content):
    content = content.replace('â', "'").replace('ð', '🎉')
    content = re.sub(r'\n+', '\n', content).strip()
    return content

# Create a Markdown file
def to_markdown(title, body):
    message_log = [
    {"role": "system", "content": "You are an AI trained to assist in generating well-formatted Markdown files. Please format the following content into a Markdown file with proper headers and styling. Any code should be formatted as code blocks. Please do not summarize the content, but rather format it as-is. Exclude headers such as 'Social Media', 'Content', 'Weeks', 'Other Resources' and any other non-root content. The root content is what I am interested in retaining. Thank you."},
    {"role": "user", "content": f"# {title}\n\n{body}"}
    ]
    formatted_content = send_message(message_log)

    # Generate a filename using the AI
    filename_message_log = [
        {"role": "system", "content": "You are an AI trained to generate a suitable filename. Please generate a filename based on the following content."},
        {"role": "user", "content": formatted_content}
    ]
    filename = send_message(filename_message_log)

    # Replace any characters that are not suitable for filenames
    filename = filename.replace(' ', '_').replace('/', '_').replace(':', '_').replace('"', '').replace('.txt', '')

    with open(f"{filename}.md", "w", encoding='utf-8') as f:
        f.write(formatted_content)

    return filename

@click.command()
@click.argument('url')
def main(url):
    html = fetch_content(url)
    title, body = parse_content(html)
    body = clean_content(body)
    filename = to_markdown(title, body)
    print(f"Content saved to {filename}.md")

# Entry point
if __name__ == "__main__":
    main()