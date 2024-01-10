#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:53:31 2023

@author: sdinesh21
"""

import fitz  # PyMuPDF

api_key = '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'

def extract_text_from_pdf(pdf_path):
    # Open the provided PDF file
    document = fitz.open(pdf_path)
    
    # Initialize a text holder
    text = ""
    
    # Loop through each page in the PDF
    for page_num in range(len(document)):
        # Get the page
        page = document.load_page(page_num)
        
        # Extract text from the page
        text += page.get_text()
    
    # Close the document
    document.close()
    
    return text

# Example usage:
# Set the path to your PDF file
pdf_path = '/Users/sdinesh21/LangChain Projects/PDF AI/bank-statement.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
# print(pdf_text)  # Print the extracted text

from langchain.embeddings import OpenAIEmbeddings

# Initialize the embeddings model (assuming you're using OpenAI's GPT)
# Make sure you have your API key set up correctly
embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)

def create_embeddings(text):
    # Create embeddings for the provided text
    return embeddings_model.embed_query(text)

# Example usage:
# Create embeddings for the extracted PDF text
pdf_embeddings = create_embeddings(pdf_text)
print(pdf_embeddings)  # Print the embeddings

from langchain.llms import OpenAI

# Initialize the language model with your OpenAI API key
lm = OpenAI(openai_api_key=api_key)

# Function to query the language model
def query_model(prompt, max_tokens=150):
    response = lm.generate(prompt, max_tokens=max_tokens)
    return response

# Function to generate a prompt based on the PDF content
def generate_prompt(pdf_text, question):
    # Here you can format the prompt in any way you want, for example:
    prompt = f"Based on the following document excerpt: \"{pdf_text[:1000]}\"... {question}"
    return prompt

# Example usage:
# Generate a prompt with a specific question
custom_prompt = generate_prompt(pdf_text, "What are the main themes discussed in this document?")
response = query_model(custom_prompt)
print("Response from the language model:", response)

# Assuming you have the 'response' from the previous steps

# Function to process the response
def process_response(response):
    # Here you can add any processing, like cleaning up the text, extracting information, etc.
    processed_text = response.strip()  # Simple example of stripping leading/trailing spaces
    return processed_text

# Example usage:
processed_response = process_response(response)
print("Processed Response:", processed_response)

# Optionally, save the response to a file or database
def save_response(response, file_path):
    with open(file_path, 'a') as file:
        file.write(response + "\n")

# Example usage:
save_response(processed_response, 'responses.txt')

# No specific code, but here are actions you might take:

# 1. Test with a variety of PDFs:
#    - Use different types of documents to see how your system performs.
#    - Pay attention to errors or unexpected outputs to refine your text extraction and prompting methods.

# 2. Refine your prompts:
#    - Based on the responses, adjust how you formulate your prompts.
#    - Try different approaches like adding more context, rephrasing questions, or specifying the type of response you're looking for.

# 3. Expand functionality:
#    - Consider adding features like summarization, more complex question answering, or integrating additional data sources.
#    - Look into Langchain's other capabilities to see what might enhance your project.

# 4. Monitor and log:
#    - Keep track of the system's performance and the types of queries and responses.
#    - Use logging to help identify patterns or areas for improvement.

# 5. Seek feedback:
#    - If possible, have others use your system and provide feedback on its outputs and usability.
#    - Use this feedback to make user-driven improvements.

# Remember to document changes and test thoroughly after each modification.
