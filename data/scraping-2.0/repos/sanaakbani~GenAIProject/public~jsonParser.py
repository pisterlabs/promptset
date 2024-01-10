import re
import PyPDF2
from prompts import PARSER_PROMPT
from aiConfig import OpenAIConfig, query_ai

# A class to parse resume PDF files and convert them into JSON format using GPT-3.5 Turbo.
class ResumeJsonParser:

    # Initialize the ResumeJsonParser with the specified configuration.
    def __init__(self, config: OpenAIConfig = OpenAIConfig(), prompt: str = PARSER_PROMPT):
        self.config = config
        self.prompt = prompt

    # Convert the PDF resume file to a JSON representation.
    def pdf2json(self, pdf_path: str):
        pdf_str = self.pdf2str(pdf_path)
        json_data = self.__str2json(pdf_str)
        return json_data

    # Convert the resume string to a JSON representation using GPT-3.5 Turbo.
    def __str2json(self, pdf_str: str):
        prompt = self.__complete_prompt(pdf_str)
        return query_ai(self.config, prompt)

    # Create a complete prompt by appending the resume string and dataset to the initial prompt.
    def __complete_prompt(self, pdf_str: str) -> str:
        return self.prompt + pdf_str

    # Convert a PDF file to a plain text string.
    def pdf2str(self, pdf_path: str) -> str:
        with open(pdf_path, "rb") as pdf_file:
            pdf = PyPDF2.PdfReader(pdf_file)
            pages = [self.__format_pdf(p.extract_text()) for p in pdf.pages]
            return "\n\n".join(pages)

    # Clean and format the PDF text string by applying pattern replacements.
    def __format_pdf(self, pdf_str: str) -> str:
        pattern_replacements = {
            r'\s[,.]': ',',
            r'[\n]+': '\n',
            r'[\s]+': ' ',
            r'http[s]?(://)?': ''
        }

        for pattern, replacement in pattern_replacements.items():
            pdf_str = re.sub(pattern, replacement, pdf_str)

        return pdf_str