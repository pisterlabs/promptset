#!/usr/bin/env python
# coding: utf-8

# In[20]:


import fitz
import numpy as np
import os
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"
key = os.environ.get("OPENAI_API_KEY")
openai.api_key = key

def extract_text_from_pdf(pdf_file_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    pdf_text = ""
    pdf_document = fitz.open(pdf_file_path)
    for page_number in range(pdf_document.page_count - 3):
        page = pdf_document.load_page(page_number)
        page_text = page.get_text("text")
        pdf_text += page_text + "\n\n"  # Add a newline after each page

    pdf_document.close()
    return pdf_text

# Extract text from a PDF file
text = extract_text_from_pdf('chapter-4.pdf')

def get_mca_questions(context: str):
    """
    Generate multiple-choice questions from a text context.

    Args:
        context (str): Text context from which questions will be generated.

    Returns:
        list: List of generated multiple-choice questions.
    """
    mcqs = []
    paragraphs = context.split("\n\n")  # Split the text into paragraphs
    # Summarize each paragraph
    for paragraph in paragraphs:
        parser = PlaintextParser.from_string(paragraph, Tokenizer('english'))
        summarizer_1 = LuhnSummarizer()
        summary_1 = summarizer_1(parser.document, 10)
        para = " "
        for sentence in summary_1:
            para += str(sentence)

        # Initialize the OpenAI language model
        llm = OpenAI(temperature=0)

        # Define a prompt template for generating questions
        template = PromptTemplate(
            input_variables=['para'],
            template="Generate multiple-choice question with choices A, B, C, and D and with more than one correct answer from the following text {para}. Also give the right answers"
        )

        # Create an LLMChain for generating questions
        chain = LLMChain(llm=llm, prompt=template, output_key="questions")
        response = chain({'para': para})
        mcqs.append(response['questions'])

    mcqs = list(set(mcqs))  # Remove duplicate questions
    return mcqs

# Get multiple-choice questions from the extracted text
questions = get_mca_questions(text)


# In[21]:


questions

