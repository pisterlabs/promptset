import PyPDF2
import openai
from common import *  # Importing the API key from the common module

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:  # Open the PDF file in binary mode
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):  # Iterate through each page
            page = reader.pages[page_num]
            text += page.extract_text()  # Extract and accumulate text from each page
        return text

def convert_to_json_with_gpt4(text):
    openai.api_key = open_ai_key  # Set the API key using the imported variable

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",  # Specify your model here
        messages=[
            {"role": "user", "content": text}
        ],
        max_tokens=4096  # Adjust the maximum token limit as needed
    )
    return response['choices'][0]['text'].strip()

# Replace the path with the actual path to your PDF file
pdf_path = r"C:\Users\thbar\OneDrive\Desktop\Document Cloud\StarterSet_Characters-part-1.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

# Print the extracted text (optional, for verification)
print("Extracted Text:\n", extracted_text)

# Convert the extracted text to JSON using GPT-4
json_output = convert_to_json_with_gpt4(extracted_text)
print("\nJSON Output:\n", json_output)
