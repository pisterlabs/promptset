import requests
from bs4 import BeautifulSoup
import openai

openai.api_key = open('openaikey.txt', 'r').read()

def fetch_web_content(url):
    """Fetches the text content of a given URL."""
    try:
        response = requests.get(url)
        response.encoding = 'utf-8'
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching content from {url}. Error: {e}")
        return ""

def chunk_text(text, chunk_size=5000):
    """Splits the text into chunks of a given size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def extract_information_from_chunk(chunk, url, data_request, final_data, progress):
    """Extracts information from a given text chunk using GPT-3 and provides a progress update."""
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    "role": "system",
                    "content": f"""
                     Here's a list of data I need to find:\n{data_request}\nRead the text the user gives you and respond only with the previous list with the information added. If the field is already full leave it or update it. Current list state:\n{final_data}"""
                },
                {"role": "user", "content": f"Reading {url}\n{chunk}"}

            ]
        )
        extracted_data = response.choices[0].message['content']
        # Add progress to the extracted data.
        return f"{progress}% Progress\n\n{extracted_data}"
    except Exception as e:
        print(f"Error extracting data from chunk. Error: {e}")
        return f"{progress}% Progress\n\n{data_request}"

def get_information_from_web(urls, data_request):
    total_chunks = sum([len(chunk_text(fetch_web_content(url))) for url in urls])
    processed_chunks = 0
    final_data = data_request

    for url in urls:
        text_content = fetch_web_content(url)
        chunks = chunk_text(text_content)

        for i, chunk in enumerate(chunks):
            processed_chunks += 1
            progress = int((processed_chunks / total_chunks) * 100)
            final_data = extract_information_from_chunk(chunk, url, data_request,final_data, progress)
            print(final_data)

    return final_data


# # Simple Test
# def test():
#     urls = ["https://en.wikipedia.org/wiki/Richard_Roundtree"]  # Replace with a real URL
#     data_request = "movies made in 1970s:\ndate of death:\n"

#     result = get_information_from_web(urls, data_request)
#     print(result)

# test()
