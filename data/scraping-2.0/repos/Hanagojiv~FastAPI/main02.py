from fastapi import FastAPI, Query
from pydantic import BaseModel
import tiktoken
from transformers import GPT2TokenizerFast
import spacy
import pandas as pd
import openai
import requests
from io import BytesIO
import time
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from config import API_KEY




app = FastAPI()


nlp = spacy.load("en_core_web_md")

df = []

def pdf_url_summary(pdf_url):
    try:
        # Download the PDF file from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()

        # Create a PDF file object from the downloaded content
        pdf_file = BytesIO(response.content)

        start_time=time.time()
        # Create a PDF reader object
        pdf_reader = PdfReader(pdf_file)

        end_time = time.time()

        # Initialize variables for summarization
        num_pages = len(pdf_reader.pages)
        total_chars = 0
        special_chars = set()
        

        # Initialize a variable to store the extracted text
        text = ''

        start_time2=time.time()
        # Iterate through each page and extract the text
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text
            
            # Update character count and collect special characters
            total_chars += len(page_text)
            special_chars.update(char for char in page_text if not char.isalnum())
        
        end_time2=time.time()

        computation_time = end_time-start_time

        computation_time2= end_time2-start_time2

        # Create a summary dictionary
        summary = {
            "Number of Pages": num_pages,
            "Total Characters": total_chars,
            "Special Characters": ", ".join(special_chars),
            "Computation time (s) for PyReader ":round(computation_time,4),
            "Computation time (s) for Extract Text":round(computation_time2,4),
        }

        return text, summary

    except Exception as e:
        return f"An error occurred: {e}"
GPT_MODEL = "gpt-3.5-turbo"
api_key = os.environ.get('API_KEY', API_KEY)
openai.api_key = api_key

class QueryRequest(BaseModel):
    question: str
    context: str

class PdfLink(BaseModel):
    pdf_url: str

@app.post("/convert_pdf")
def convert_pdf(pdf_link: PdfLink):
    pdf_url = pdf_link.pdf_url
    global context
    
    text, summary = pdf_url_summary(pdf_url)
    
    return {"text": text, "summary": summary}


def num_tokens(text):
    tokenizer = GPT2TokenizerFast.from_pretrained(GPT_MODEL)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return len(encoding)

def query_message(query, text,  token_budget):
    strings, relatednesses = strings_ranked_by_relatedness(query, text)
    introduction = 'Use the relevant documents from the SEC government data to answer the subsequent question. If the answer cannot be found in the documents, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    remaining_budget = token_budget
    for string in strings:
        next_document = f'\n\nSEC Document Section:\n"""\n{string}\n"""'
        if num_tokens(message + next_document + question) > remaining_budget:
            message += question
            
            break
        else:
            message += next_document
    return message + question

# def strings_ranked_by_relatedness(query, text, top_n=100):
#     query_embedding = openai.Embed.create(model=GPT_MODEL, data=query)
#     text_embeddings = openai.Embed.create(model=GPT_MODEL, data=text)
#     # similarities = [query_embedding.similarity(embedding) for embedding in text_embeddings]
#     similarities = [cosine_similarity(query_embedding['embeddings'][0], embedding['embeddings'][0]) for embedding in text_embeddings['embeddings']]
#     sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    
#     top_strings = [text[i] for i in sorted_indices[:top_n]]
#     top_relatednesses = [similarities[i] for i in sorted_indices[:top_n]]
    
#     return top_strings, top_relatednesses
# def create_embedding(context):
#     try:
#         response = openai.Embedding.create(
#             model= "text-embedding-ada-002",
#             input=context
#         )
#         embeddings = [item['embedding'] for item in response['data']]
#         flattened_embeddings = [value for sublist in embeddings for value in sublist]
#         return flattened_embeddings
#     except Exception as e:
#         print(f"Error in generating Embedding: {e}")
#         return ""


def strings_ranked_by_relatedness(query, text, top_n=100):
    query_embedding = nlp(query).vector
    text_embeddings = text_embeddings = [nlp(section).vector for section in text]
    similarities = [cosine_similarity(query_embedding, embedding) for embedding in text_embeddings]

    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

    top_strings = [text[i] for i in sorted_indices[:top_n]]
    top_relatednesses = [similarities[i] for i in sorted_indices[:top_n]]

    return top_strings, top_relatednesses


def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = sum(a * a for a in vector1) ** 0.5
    magnitude2 = sum(b * b for b in vector2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)


def generate_context_from_summary(summary: str):
 
    sections = summary.split("\n") 


    filtered_sections = [section for section in sections if len(section.split()) > 20]

    if filtered_sections:
        
        data = {
            "text": filtered_sections
        }

        df = pd.DataFrame(data)

        
        print("Filtered Sections:")
        print(df)

        
        context = "\n".join(filtered_sections)

        return context
    else:
        print("No filtered sections found in the summary.")
        return "Unable to extract text from the PDF"

def generate_context_from_summary_nougat(summary: str):
    # Split sections based on double newline characters
    sections = re.split(r'\n\n', summary)
    
    # Filter sections that have more than 20 words (adjust as needed)
    filtered_sections = [section for section in sections if len(section.split()) > 40]

    if filtered_sections:
        data = {
            "text": filtered_sections
        }

        # Create a DataFrame for better visualization
        df = pd.DataFrame(data)

        print("Filtered Sections:")
        print(df)

        # Join the filtered sections to create context
        context = "\n".join(filtered_sections)

        return context
    else:
        print("No filtered sections found in the summary.")
        return "Unable to extract text from the PDF"

class SummaryRequest(BaseModel):
    summary: str
    
@app.post("/data-collection")
def data_collection(request_data: SummaryRequest ):
    global  context 
    context = generate_context_from_summary(request_data.summary)
    return {"context": context}

@app.post("/data-collection_nougat")
def data_collection(request_data: SummaryRequest ):
    global  context 
    context = generate_context_from_summary_nougat(request_data.summary)
    return {"context": context}

@app.post("/ask")
def ask_question(query_request: QueryRequest):
    query = query_request.question
    context = query_request.context
    token_budget = 4096 - 500
    message = query_message(query, context, token_budget)
    messages = [
        {"role": "system", "content": "You answer questions about SEC government data."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(model=GPT_MODEL, messages=messages, temperature=0)
    response_message = response["choices"][0]["message"]["content"]
    print(df)
    return {"answer": response_message}


if __name__ == "__main__":
    import os

# Disable tokenizers parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)