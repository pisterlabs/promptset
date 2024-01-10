# import requests
# from bs4 import BeautifulSoup
# import os
# from dotenv import load_dotenv
# import openai
# from urllib.parse import urljoin
# import json
# import random
# import time

# # Load environment variables from .env file
# load_dotenv()

# # Access the API key from the environment variable
# openai_api_key = os.getenv('OPENAI_API_KEY')
# openai.api_key = openai_api_key

# def download_via_api(api_endpoint, save_folder):
#     response = requests.get(api_endpoint)
#     data = response.json()
#     print(f"Downloaded data from {api_endpoint}")
#     print(f"Saving data to {save_folder}")

# def api_call2(messages, functions1, max_response_tokens=100):
#     for i in range(15):
#         try:
#             return openai.ChatCompletion.create(
#                 model="gpt-4-0613",
#                 messages=messages,
#                 functions=functions1,
#                 temperature=0.7,
#                 max_tokens=max_response_tokens,
#                 function_call={"name": "find_tags_class"}
#             )
#         except openai.error.RateLimitError as e:
#             print(f"Rate limit exceeded: {e}")
#             wait_time = 2 ** i + random.random()
#             print(f"Waiting for {wait_time} seconds before retrying...")
#             time.sleep(wait_time)
#         except Exception as e:
#             print(f"An unexpected error occurred: {e if isinstance(e, str) else repr(e)}")
#             raise
#     print("Maximum number of retries exceeded. Aborting...")

# def ask_openai_for_tags(html_content):
#     messages = [
#         {
#             "role": "system",
#             "content": "You are an expert in web scraping. Given an HTML content, identify the tags or sections that seem to be links to the documentation."
#         },
#         {
#             "role": "user",
#             "content": f"Here's the HTML content: {html_content[:12000]}... (truncated for brevity). Please suggest the relevant tags or sections for scraping. include links/hiperlinks and codeboxes"
#         }
#     ]
#     functions1 = [{
#         "name": "find_tags_class",
#         "description": "Scrape the web using beautiful soup using tags and classes for scraping.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "tags": {
#                     "type": "array",
#                     "items": {
#                         "type": "string"
#                     },
#                     "description": "The HTML tags."
#                 },
#                 "classes": {
#                     "type": "array",
#                     "items": {
#                         "type": "string"
#                     },
#                     "description": "The classes of the HTML tags."
#                 }
#             },
#             "required": ["tags", "classes"]
#         }
#     }]
    
#     completion = api_call2(messages, functions1)
#     reply_content = completion.choices[0]
#     args = reply_content["message"]['function_call']['arguments']
#     args = json.loads(args)
#     tags = args["tags"]
#     classes = args["classes"]
#     return tags, classes

# def extract_content(html_content, tags, classes):
#     soup = BeautifulSoup(html_content, 'html.parser')
#     elements = [soup.find_all(tag, class_=class_name) for tag, class_name in zip(tags, classes)]
#     elements = [item for sublist in elements for item in sublist]
#     extracted_texts = [element.get_text(strip=True) for element in elements]
#     extracted_links = [link['href'] for link in elements if link.has_attr('href')]
#     return extracted_texts, extracted_links

# # def download_via_scraping(url, save_folder):
# #     if not os.path.exists(save_folder):
# #         os.makedirs(save_folder)
# #     response = requests.get(url)
# #     response.raise_for_status()
# #     print(f"Downloaded HTML content from {url}")
# #     tags, classes = ask_openai_for_tags(response.text)
    
# #     print(f"Suggested tags: {tags}")
# #     print(f"Suggested classes: {classes}")
    
# #     extracted_texts, doc_links = extract_content(response.text, tags, classes)
    
# #     print(f"Extracted links: {doc_links}")
    
# #     filename = os.path.join(save_folder, f"extracted_texts_from_{os.path.basename(url)}.txt")
# #     with open(filename, 'w', encoding='utf-8') as f:
# #         for text in extracted_texts:
# #             f.write(text + '\n\n')
# #     print(f"Saved extracted texts to {filename}")
# #     for doc_url in doc_links:
# #         full_url = urljoin(url, doc_url)
# #         try:
# #             doc_response = requests.get(full_url)
# #             subpage_texts, _ = extract_content(doc_response.text, tags, classes)
# #             filename = os.path.join(save_folder, f"extracted_texts_from_{os.path.basename(doc_url)}.txt")
# #             with open(filename, 'w', encoding='utf-8') as f:
# #                 for text in subpage_texts:
# #                     f.write(text + '\n\n')
# #             print(f"Saved extracted texts from subpage {full_url} to {filename}")
# #         except Exception as e:
# #             print(f"Error processing {full_url}: {e}")


# def download_via_scraping(url, save_folder):
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     response = requests.get(url)
#     response.raise_for_status()
#     print(f"Downloaded HTML content from {url}")
#     tags = [("div", {"role": "main"}), ("main", {"id": "main-content"})]
#     classes = ask_openai_for_tags(response.text)
    
#     print(f"Suggested tags: {tags}")
#     print(f"Suggested classes: {classes}")
    
#     extracted_texts, doc_links = extract_content(response.text, tags, classes)
    
#     print(f"Extracted links: {doc_links}")
    
#     filename = os.path.join(save_folder, f"extracted_texts_from_{os.path.basename(url)}.txt")
#     with open(filename, 'w', encoding='utf-8') as f:
#         for text in extracted_texts:
#             f.write(text + '\n\n')
#     print(f"Saved extracted texts to {filename}")
#     for doc_url in doc_links:
#         full_url = urljoin(url, doc_url)
#         try:
#             doc_response = requests.get(full_url)
#             subpage_texts, _ = extract_content(doc_response.text, tags, classes)
#             filename = os.path.join(save_folder, f"extracted_texts_from_{os.path.basename(doc_url)}.txt")
#             with open(filename, 'w', encoding='utf-8') as f:
#                 for text in subpage_texts:
#                     f.write(text + '\n\n')
#             print(f"Saved extracted texts from subpage {full_url} to {filename}")
#         except Exception as e:
#             print(f"Error processing {full_url}: {e}")
# def download_documentation(url, save_folder):
#     if 'api' in url:
#         download_via_api(url, save_folder)
#     else:
#         download_via_scraping(url, save_folder)

# download_documentation('https://www.cyberbotics.com/doc/reference/index', '/home/stahlubuntu/coder_agent/extractt')



from langchain.document_loaders import ReadTheDocsLoader

loader = ReadTheDocsLoader("rtdocs", features="html.parser")

docs = loader.load()

print(docs)