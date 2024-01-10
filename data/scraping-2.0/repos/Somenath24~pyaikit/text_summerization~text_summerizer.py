# Import the required libraries
import PyPDF2
import requests
import PySimpleGUI as sg
import os
import openai

# Define a class for the PDF Summarizer
class text_summerizer:
    # Define the instance variables for the class
    def __init__(self):
        self.pdf_file = None
        self.pdf_reader = None
        self.num_pages = 0
        self.start_page = 0
        self.end_page = 0
        self.text = ''
        self.summary=''

    # Open the PDF file
    def open_pdf_file(self, file_path):
        self.pdf_file = open(file_path, 'rb')
        self.pdf_reader = PyPDF2.PdfReader(self.pdf_file)
        self.num_pages = len(self.pdf_reader.pages)
        
    # Close the PDF file
    def close_pdf_file(self):
        if self.pdf_file is not None:
            self.pdf_file.close()
        self.pdf_reader = None
        self.num_pages = 0

    # Set the starting and ending page numbers
    def set_start_page(self, start_page):
        end_page=start_page.split("-")[1]
        start_page=start_page.split("-")[0]
        if(start_page==''):
            start_page=1
        self.start_page = int(start_page) - 1
        self.end_page = int(end_page) - 1

    # Extract the text from the selected pages of the PDF
    def extract_text_func(self):
        self.text = ''
        for page_num in range(self.start_page, self.end_page):
            page = self.pdf_reader.pages[page_num]
            self.text += page.extract_text()
            #self.summary += self.summarize_text(page.extract_text())

    # Run the OpenAI summarization model
    def call_openai(self, openai, prompt):

        # Set the model parameters and generate the summary using OpenAI API
        response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt=prompt,
        temperature=0.8,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

        if 'choices' in response:
            if len(response['choices']) > 0:
                ret = response['choices'][0]['text']
            else:
                ret = 'No responses'
        else:
            ret = 'No responses'

        return ret
    
    # Summarize the text using the OpenAI model
    def summarize_text(self, openai, text, no_of_words):
        """
        Generate a summary of the given text using an AI model.

        Args:
            self (object): The instance of the class.
            openai (object): The OpenAI object for accessing the API.
            text (str): The text to be summarized.
            no_of_words (int): The desired number of words in the summary.

        Returns:
            str: The generated summary of the text.

        """

        # Set the prompt for the summarization model
        prompt = f'Summarize the following text in a maximum of {no_of_words} words:\n{text}'

        try:
            # Call the call_openai method to generate the summary
            summary = self.call_openai(openai, prompt)
        except Exception as e:
            summary = str(e)
            # Handle exception, such as printing an error message or raising further exceptions
            # print(f'Exception: {str(e)}')

        return summary.strip()
    
     # Summarize the text using the OpenAI model
    def summarize_pdf(self, openai, no_of_words, file_path, start_page=1, end_page=None):
        """
        Summarize a PDF file by extracting text from specified pages and generating a summary.

        Args:
            self (object): The instance of the class.
            openai (object): The OpenAI object for accessing the API.
            no_of_words (int): The desired number of words in the summary.
            file_path (str): The path to the PDF file to be summarized.
            start_page (int, optional): The starting page number for text extraction (default is 1).
            end_page (int, optional): The ending page number for text extraction (default is None, i.e., extract till the last page).

        Returns:
            str: The generated summary of the PDF file.

        """

        # Open the PDF file
        self.open_pdf_file(file_path=file_path)

        # Set the start and end page for text extraction
        self.start_page = start_page - 1  # Adjusting to 0-based index
        self.end_page = end_page

        # Extract the text from the specified pages
        self.extract_text_func()

        # Generate a summary of the extracted text
        summary_text = self.summarize_text(openai, self.text, no_of_words)

        # Close the PDF file
        self.close_pdf_file()

        return summary_text


    
 