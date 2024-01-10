import os, sys
import openai
import time
from bs4 import BeautifulSoup
import requests

# --- Markdown Header: Initialization and Environment Setup ---

openai.api_key = os.getenv("OPENAI_API_KEY")

if len(sys.argv) > 1:
    rtf_file = sys.argv[1]
else:
    rtf_file = "IMF/22 USC Chapter 7 IMF subscription information.rtf"

url_regex = 'HYPERLINK "(http[^"]*)"'
urls = []

# --- Markdown Header: Reading and Filtering URLs ---

with open(rtf_file, 'r') as file:
    for line in file:
        if 'HYPERLINK' in line:
            url = line.split('"')[1]
            urls.append(url)

cornell_urls = [url for url in urls if 'cornell' in url]

title_permutations=["#page_title", "#page-title"]
page_title = ""
# --- Markdown Header: HTML Element Extraction ---

def extract_elements(html):
    soup = BeautifulSoup(html, 'html.parser')

    for title_permutation in title_permutations:
        page_title_elem = soup.select_one(title_permutation)
        if page_title_elem:
            title_id = title_permutation
            break

    page_title = page_title_elem.text
    page_title_parent = page_title_elem.parent
    page_title_element = page_title_parent.select_one(title_id)

    tab_content_elements = page_title_element.find_next_siblings()
    tab_content = ''
    for element in tab_content_elements:
        tab_content += element.text + "\n"

    return f"## {page_title}\n\n", tab_content

MODEL = "gpt-4"

# --- Markdown Header: API Calls and Responses ---


for url in cornell_urls:
    response = requests.get(url)
    output = response.text

    try:
        page_title, tab_content = extract_elements(output)
    except Exception as e:
        print(f"Error during request: {e}", file=sys.stderr)
        print(f"URL: {url}", file=sys.stderr)
        print(f"press enter for output dump or ctrl-c", file=sys.stderr)
        input()
        print(f"Output: {output}", file=sys.stderr)
        exit(1)

    print(page_title)
    print(tab_content)

    prompt = "\n\n" + page_title + "\nContent: " + tab_content
    prompt = prompt[:10000]
    prompt1 = prompt[:prompt.rfind(' ')]

    while True:
        try:
            api_response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "user",
                     "content": prompt1 + "\n(skeptical and mildly jaded analogies and summaries follow, with any significant timeline events or commonly referred counterparts)"},
                ],
                temperature=0.8,
            )
            break
        except Exception as e:
            print(f"Error during request: {e}", file=sys.stderr)
            print("Waiting for 5 seconds before retrying...", file=sys.stderr)
            time.sleep(5)

    # print(f"\n\n### commentary on {page_title}")
    # print(f"\n\n### commentary on {page_title}")
    print(f"\n\n### commentary on {page_title}") #this doesn't expand at runtime for some reason



    print(api_response.choices[0].message.content.strip())
    print("\n\n---\n")