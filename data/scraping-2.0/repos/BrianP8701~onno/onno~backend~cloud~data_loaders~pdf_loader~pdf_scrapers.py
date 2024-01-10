'''
This class contains functions that scrape text from PDFs.

I want this to be an automatic part of the pipeline, just pass in
a path to a pdf, and get back strings.

After using each a little bit, I found, annoyingly, that each of these
sometimes might not work well on some pdfs for no seemingly good reason.
Issues, like mashing together words, interpreting pictures or math equations
as a bunch of weird characters, adding spaces between each letter, etc.

To get around this, I just use all three of them on a pdf, and then
ask ChatGPT to look at samples from each of them and pick the one that
looks the best. Of course, in the rare case where all three of them
break down, your going to have some ugly data.
'''
import pdfplumber
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import random
import json
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.high_level import extract_pages
from typing import List
from openai_utils import choose_best_scraped_text

class PDFLoader():
    def __init__(self):
        self.scrapers = [
            self.pdfplumber_scraper,
            self.pypdf2_scraper,
            self.pytesseract_scraper,
            self.pdfminer_scraper
        ]

    def scrape(self, uploaded_file):
        '''
        Extract text from a PDF using all available scrapers and have ChatGPT pick the best extraction.
        
        Returns a list of strings, each string is a page of text.
        '''
        all_text = []  # Store results from each scraper
        for scraper in self.scrapers:
            try:
                extracted_text = scraper(uploaded_file)
                all_text.append(extracted_text)
            except Exception as e:
                raise Exception(f'{scraper} failed: {e}')

        # Collect random samples from each scraper result for comparison
        samples = []
        rand_chunk_indices = [random.randint(0, len(all_text[0])-1) for _ in range(3)]
        for scraper_index, scraped_chunks in enumerate(all_text, 1):
            sample = ' '.join([scraped_chunks[i][:100] for i in rand_chunk_indices])
            sample = f'<Sample {scraper_index}>: {sample}\n'
            samples.append(sample)

        # Ask ChatGPT to determine the best sample
        best_sample_index = self.choose_best_scraped_text(samples)
        print(best_sample_index)
        return all_text[best_sample_index]

    def pdfminer_scraper(self, uploaded_file):
        extracted_text = []
        for page_layout in extract_pages(uploaded_file):
            page_text = []
            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    page_text.append(element.get_text())
            extracted_text.append(' '.join(page_text))
        return extracted_text

    def pdfplumber_scraper(self, uploaded_file):
        all_text = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                all_text.append(text)
        return all_text

    def pypdf2_scraper(self, uploaded_file):
        pdf_reader = PdfReader(uploaded_file)
        text_list = []
        for page in pdf_reader.pages:
            text_content = page.extract_text()
            text_list.append(text_content)
        return text_list

    def pytesseract_scraper(self, uploaded_file):
        # Read the file content into a byte array
        file_content = uploaded_file.read()
        
        # Use convert_from_bytes instead of convert_from_path
        imgs = convert_from_bytes(file_content)
        
        chunks = []
        for img in imgs:
            chunks.append(pytesseract.image_to_string(img))
        return chunks