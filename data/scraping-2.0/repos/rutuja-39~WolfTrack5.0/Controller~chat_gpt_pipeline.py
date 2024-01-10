'''
MIT License

Copyright (c) 2023 Shonil B, Akshada M, Rutuja R, Sakshi B

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from PyPDF2 import PdfFileReader
import numpy as np
import requests
import os
import openai


def pdf_to_text(pdf_path, text_path):
    with open(pdf_path, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PdfFileReader(pdf_file)

        # Initialize an empty string to store the text
        text = ""

        # Iterate through each page of the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Extract text from the current page
            text += pdf_reader.pages[page_num].extract_text()

        # Open a text file in write mode and save the extracted text
        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)


# chatgpt pipeline
def chatgpt(resume_textfile_path):
    with open(resume_textfile_path, 'r') as file:
        content = file.read()
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Set the API endpoint
    api_url = "https://api.openai.com/v1/chat/completions"

    # Set the request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # Set the request payload
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role":"system","content":"critique the following resume text on the basis of its conciseness, use of action words and numbers. Give suggestions on the 1. education section, then the 2. experiences section, then the 3. skills section and finally the 4. projects section. Give these suggestions on these four sections in the form of four paragraphs and label them Section 1, Section 2, Section 3 and Section 4 respectively, each separated by a line. Make sure each paragraph is atleast 50-70 words long."},{"role": "user", "content": content}],
        "temperature": 0.7
    }

    # Make the POST request
    try:
        response = requests.post(api_url, json=payload, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Print or process the response content
            print("API Response:")
            json_data = response.json()
            final_suggestions = json_data['choices'][0]['message']['content']
            # print(final_suggestions)
            return final_suggestions
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")


