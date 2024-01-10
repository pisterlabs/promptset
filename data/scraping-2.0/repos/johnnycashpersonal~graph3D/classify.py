import os
import openai
import pdfplumber
from tkinter import Tk, filedialog

openai.organization = "org-ge4VJKT4YPXVCf7EpXv2H7NZ"
openai.api_key_path = "C:\\Users\\johnm\\dev\\Graph3D\\\key.txt"

def open_file_dialog():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path

def ocr_pdf_with_tesseract(pdf_file_path):
    # Open the PDF
    with pdfplumber.open(pdf_file_path) as pdf:
        # Extract text from the first page
        first_page = pdf.pages[3]
        text = first_page.extract_text()
    print(text)
    return text

def ask_gpt3_about_paper(text):
    messages = [
        {"role": "user", "content": f' This is a research paper, which always contains a title (generally the first text inputs) and an "Abstract"- The papers overview (which usually comes right after the title). You MUST find both of these from within the following text block and print them out: {text}. Create a list of keywords focused both on what was done in the paper (at least from what you know based on the abstract) and why it was done. These keywords should be in a comma-seperated list. Do not number the keywords. Do not describe the keywords, simply list what they are. Also, generate a short, one or two-sentence description of what this paper could mean for future AI capabilities and improvements, labelled as "Impact.'}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def main():
    pdf_file_path = open_file_dialog()
    if pdf_file_path:
        text = ocr_pdf_with_tesseract(pdf_file_path)
        gpt3_response = ask_gpt3_about_paper(text)
        print(gpt3_response)
        return
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()
