

import json
import requests
from bs4 import BeautifulSoup
import openai
import time
import random
from dotenv import load_dotenv
import os



# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key



def api_call(messages, max_response_tokens):
    for i in range(15):
        try:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages,
                
                temperature=0.2,
                max_tokens=max_response_tokens,
                
            )
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            wait_time = 2 ** i + random.random()
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"An unexpected error occurred: {e if isinstance(e, str) else repr(e)}")
            raise
    print("Maximum number of retries exceeded. Aborting...")

def summarize_text(text):
    """
    Summarizes text using the OpenAI API.

    Parameters:
        text (str): The text to summarize.

    Returns:
        str: The summarized text.
    """



    messages = [
        {
            "role": "system",
            "content": "You are a sophisticated AI that has the ability to summarize chunks of text. Please provide a detailed summary of the following text that focuses on the code-related content. Your summary should include all code-related text, including code examples and explanations. Please make sure to highlight the most important aspects of the code and provide a clear and concise summary that is easy to understand for a technical audience. Please avoid summarizing non-code-related content such as introductions, conclusions, or irrelevant details. Your summary should be written in a clear and concise manner, and should be easy to understand for a technical audience. Please take your time to carefully read and understand the text before summarizing it. If you encounter any technical terms or jargon that may be unfamiliar to a general audience, please provide a brief explanation or definition."
        },
        {
            "role": "user",
            "content": text
        }
    ]

    response = api_call(messages=messages, max_response_tokens=600)
    summary = response["choices"][0]["message"]["content"]
    return summary





def scrape_web_pages(urls):
    """
    Scrapes text content from a list of web pages, chunks the text into 40,000 character chunks if necessary, summarizes each chunk using the OpenAI API, and combines the summarized chunks into a single string. Then, passes all the chunks to another API call to generate a final summary if the text has been chunked.

    Parameters:
        urls (list): A list of URLs to scrape.

    Returns:
        str: A JSON-formatted string containing the URL and its summarized text.
    """
    
    scraped_data = []
    
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text()
            
            # Chunk the text into 40,000 character chunks if necessary
            if len(text_content) > 40000:
                chunks = [text_content[i:i+40000] for i in range(0, len(text_content), 40000)]
                
                # Summarize each chunk using the OpenAI API
                summarized_chunks = [summarize_text(chunk) for chunk in chunks]
                # Join the summarized chunks into a single string
                summarized_chunks = "\n".join(summarized_chunks)
                
 
                messages = [
                    {
                        "role": "system",
                        "content": "You are a sophisticated AI that has the ability to generate a final summary based on chunks of text that were summarized individually. The original text was too long to summarize in one go, so it was chunked into smaller pieces and each piece was summarized separately. Please generate a final summary that is focused on the code-related content for the following chunks:"
                    },
                    {
                        "role": "user",
                        "content": summarized_chunks
                    }
                ]
                response = api_call(messages=messages, max_response_tokens=600)
                final_summary = response["choices"][0]["message"]["content"]
                
            else:
                # Summarize the text using the OpenAI API
                final_summary = summarize_text(text_content)
                
            scraped_data.append({
                "url": url,
                "summarized_text": final_summary
            })
            
        except Exception as e:
            scraped_data.append({
                "url": url,
                "error": str(e)
            })
            
    # Convert the list of dictionaries to a JSON-formatted string
    scraped_data_str = json.dumps(scraped_data)
    
    return scraped_data_str


# # Example Usage

# urls_to_scrape = [
#     "https://openai.com/blog/function-calling-and-other-api-updates?ref=upstract.com"
# ]

# scraped_results_str = scrape_web_pages(urls_to_scrape)
# print(scraped_results_str)


