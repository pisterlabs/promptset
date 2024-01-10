import boto3
import requests
import time
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
from WeaviateClient import add_item, add_batch_items
import openai
import tracemalloc
tracemalloc.start()
import functools


def get_chunks_with_overlap(text, chunk_size=1000, overlap_size=150):
    """Splits the text into chunks of specified size with overlap."""
    chunks = []
    index = 0

    while index < len(text):
        # If we are not at the start, move back to create overlap
        if index > 0 and (index - overlap_size) > 0:
            index -= overlap_size

        # Find the end of the chunk
        end = min(index + chunk_size, len(text))

        # Ensure the chunk ends on a space (if not the end of the text)
        if end < len(text):
            while end > index and text[end] != ' ':
                end -= 1

        # Add the chunk
        chunks.append(text[index:end].strip())

        # Move the index forward
        index = end

    return chunks

# Function to process a single page
async def process_page(executor, client, image):
    loop = asyncio.get_event_loop()
    with io.BytesIO() as image_bytes:
        image.save(image_bytes, format='JPEG')
        image_bytes_val = image_bytes.getvalue()

    # Prepare the function call with functools.partial
    func = functools.partial(client.detect_document_text, Document={'Bytes': image_bytes_val})

    # Call Textract using run_in_executor
    response = await loop.run_in_executor(executor, func)
    text = '\n'.join([block['Text'] for block in response['Blocks'] if block['BlockType'] == 'LINE'])

        
    return text

def get_title(text, model="gpt-3.5-turbo"):

    try:
        response = openai.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "user", "content": f"Extract the title of the paper from the given context. \
                                    Note that it may be in multiple lines. Do not make any assumptions.\nContext:\"\"\"{text}\"\"\""}],
                    max_tokens=1000,
                    temperature=0.0)
        extracted_title = response.choices[0].message.content
        return extracted_title
    
    except:
        try:
            # assuming exception is due to too many tokens
            # chunk the text and choose first 3000 tokens
            text = get_chunks_with_overlap(text=text, chunk_size=3000,overlap_size=0)[0]
            response = openai.chat.completions.create(
                        model=model,
                        messages=[
                        {"role": "user", "content": f"Extract the title of the paper from the given context. \
                                        Note that it may be in multiple lines. Do not make any assumptions.\nContext:\"\"\"{text}\"\"\""}],
                        max_tokens=1000,
                        temperature=0.0)
                    
            extracted_title = response.choices[0].message.content
            return extracted_title
        
        except Exception as e:
            print("Error finding the title: ", e)
            return "" # currently not returning any title on exception

async def load_pdf_with_textract(class_name, properties=None):
    start_time = time.time()
    url = properties["url"]    

    response = requests.get(url)
    with open('document.pdf', 'wb') as file:
        file.write(response.content)

    # Convert PDF to images
    images = convert_from_path('document.pdf')

    # Setup AWS Textract client
    session = boto3.Session()
    client = session.client('textract')

    # Setup thread pool for running I/O tasks in parallel within a context manager
    with ThreadPoolExecutor(max_workers=len(images)) as executor:
        tasks = [process_page(executor, client, image) for image in images]
        results = await asyncio.gather(*tasks)

    # time to get the text from the images
    images_time = time.time()
    print("time to get the text from the images: " + str(images_time - start_time))

    combined_pages = []
    overlap_size = 150
    for i in range(len(results)):
        combined_text = results[i]
        if i < len(results) - 1:  # if not the last page, append the beginning of the next page
            combined_text += "\n" + results[i + 1][:overlap_size]  # overlap_size from get_chunks_with_overlap
        combined_pages.append((i + 1, combined_text))  # Store page number with combined text

    batch_items = []  # Initialize a list to collect batch items
    for page_number, text in combined_pages:

        # extract title from the first page
        if page_number == 1:
            extracted_title =  get_title(text)

        text_chunks = get_chunks_with_overlap(text)
        for chunk in text_chunks:
            modified_properties = properties.copy()
            modified_properties["page_number"] = str(page_number)
            modified_properties["text"] = chunk
            modified_properties["title"] = extracted_title
            batch_items.append(modified_properties)  # Add item to the batch list

            # When batch size is reached, send the batch
            if len(batch_items) >= 100:
                add_batch_items(class_name, batch_items)
                batch_items = []  # Reset the batch list after sending
    
    if batch_items:
        add_batch_items(class_name, batch_items)  # Send the remaining batch

    end_time = time.time()
    print("time elapsed: " + str(end_time - start_time))
    print("uploaded file to class: " + class_name + " with path: " + properties["path"] + ", title: " + 
          modified_properties["title"] + ", and url: " + properties["url"])

    return "Success"
# import pypdf
# from pdf2image import convert_from_path
# import pytesseract
# import json
# import os
# import requests
# import datetime
# from utils import utils
# import tempfile
# import time
# from WeaviateClient import add_item

# import concurrent.futures

# def load_pdf(class_name, properties=None):
#     print(properties)
#     start_time = time.time() 
#     try:
#         url = properties["url"]
#         print("loading pdf: " + url + "...")
#         # load file from a given url
#         response = requests.get(url)
#         print(response)
#         response.raise_for_status()
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(response.content)


#         with open(tmp_file.name, "rb") as pdf_file:
#             pdf_reader = pypdf.PdfReader(pdf_file)
#             print("file loaded")
#             num_pages = len(pdf_reader.pages)
#             pages_text = []
#             pageCounter = 0
#             print("file has " + str(num_pages) + " pages")

#             def process_page(page):
#                 print("reading page: " + str(page + 1) + "...")
#                 local_path = os.path.abspath(tmp_file.name)
#                 images = convert_from_path(
#                     local_path, first_page=page + 1, last_page=page + 1)
#                 # if there are images in the page, use OCR to extract text
#                 if images:
#                     page_image = images[0]
#                     page_text = pytesseract.image_to_string(page_image)
#                     page_text = ""
#                     # pages_text.append(page_text)
#                 # if there are no images in the page, use PyPDF2 to extract text
#                 else:
#                     print("no images found in page " + str(page + 1) + "...")
#                     page_obj = pdf_reader.getPage(page)
#                     page_text = page_obj.extractText()
#                     pages_text.append(page_text)

#                 # print("page " + str(page + 1) + ": " + page_text)

#                 # split text into into chunks of 1000 characters when the word ends
#                 text_chunks = utils.get_chunks(page_text)

#                 for chunk in text_chunks:
#                     modified_properties = properties.copy()
#                     modified_properties["page_number"] = str(page)
#                     modified_properties["text"] = chunk

#                     # add_item(class_name=class_name, item=modified_properties)
                    

#             # parallelize the process_page function
#             # with concurrent.futures.ThreadPoolExecutor() as executor:
#             #     executor.map(process_page, range(num_pages))

#             for page in range(num_pages):
#                 process_page(page)
                
#             pageCounter += num_pages
            
#         # end timer
#         end_time = time.time()
#         print("time elapsed: " + str(end_time - start_time))
#         return "Success"
#     except Exception as e:
#         print("Error loading pdf:", e)
#         return "Failure"


# async def main():
#     url_8page = "https://emoimoycgytvcixzgjiy.supabase.co/storage/v1/object/sign/documents/23c88506-31e5-43c7-911c-d6df61fbbf7b/curve-stablecoin.pdf?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJkb2N1bWVudHMvMjNjODg1MDYtMzFlNS00M2M3LTkxMWMtZDZkZjYxZmJiZjdiL2N1cnZlLXN0YWJsZWNvaW4ucGRmIiwiaWF0IjoxNzAzOTA4MzA3LCJleHAiOjE3MDQ1MTMxMDd9.PVXyAmoZqWlrSt2-v5ma6P9oZrFlm-7vqTSytAAkcNo&t=2023-12-30T03%3A51%3A47.332Z"
#     url_29page = "https://emoimoycgytvcixzgjiy.supabase.co/storage/v1/object/sign/documents/06ca3fba-93e4-493d-9c22-5c5ddc15d352/G3-2023-404312_Merged_PDF.pdf?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJkb2N1bWVudHMvMDZjYTNmYmEtOTNlNC00OTNkLTljMjItNWM1ZGRjMTVkMzUyL0czLTIwMjMtNDA0MzEyX01lcmdlZF9QREYucGRmIiwiaWF0IjoxNzAzOTA4Mjc3LCJleHAiOjE3MDQ1MTMwNzd9.3CJFZeo6s7XchyaWmyD-6rkxU-JqnQPulZfgLOc5KB8&t=2023-12-30T03%3A51%3A17.288Z"
    
#     # short_1706.03762.pdf
#     # url_attn_short = 'https://emoimoycgytvcixzgjiy.supabase.co/storage/v1/object/sign/documents/c75767dd-172c-463c-aafc-1e2dfddc1b32/yoho/Folder%20X/Folder%20X/short_1706.03762.pdf?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJkb2N1bWVudHMvYzc1NzY3ZGQtMTcyYy00NjNjLWFhZmMtMWUyZGZkZGMxYjMyL3lvaG8vRm9sZGVyIFgvRm9sZGVyIFgvc2hvcnRfMTcwNi4wMzc2Mi5wZGYiLCJpYXQiOjE3MDM5MTc4OTIsImV4cCI6MTcwMzkxNzk1Mn0.iL-8ARX2INdXIy5XtvIQYAXsmUhvD-er4Kb6y1abJTc'
#     url_4page = "https://emoimoycgytvcixzgjiy.supabase.co/storage/v1/object/sign/documents/539e9941-5673-4561-8f7b-ddb523a4b537/Test/Additional/CVF_2024_short.pdf?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJkb2N1bWVudHMvNTM5ZTk5NDEtNTY3My00NTYxLThmN2ItZGRiNTIzYTRiNTM3L1Rlc3QvQWRkaXRpb25hbC9DVkZfMjAyNF9zaG9ydC5wZGYiLCJpYXQiOjE3MDQ0Njc0OTQsImV4cCI6MTczNjAwMzQ5NH0.m0rbeiCSkHAfOzqN3t6IDFvoT386og90m45J8O6nLw4&t=2024-01-05T15%3A11%3A35.234Z"
#     result = await load_pdf_with_textract(class_name="File_store_ver2", properties={"path": "abcd/cvf.pdf", "url": url_4page})
#     print(result)



# # Running the main function
# if __name__ == "__main__":
#     asyncio.run(main())

