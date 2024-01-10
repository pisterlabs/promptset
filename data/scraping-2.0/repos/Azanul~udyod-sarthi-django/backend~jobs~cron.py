from time import sleep
import openai
import requests
import pinecone
import PyPDF2
import hashlib
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
from .models import Job

base = "https://niepmd.tn.nic.in"
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "b3ea86b7-3555-4689-b27b-0a05a336a8a7"

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY)
index = pinecone.Index(index_name="pdf_embeddings")

def get_pdf_links(url):
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
    return pdf_links

def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url, verify=False)
    with open(save_path, 'wb') as file:
        file.write(response.content)
    
def get_pdf_content_hash(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_content = file.read()
        return hashlib.sha256(pdf_content).hexdigest()

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def store_pdf_embedding(pdf_path):
    pdf_hash = get_pdf_content_hash(pdf_path)
    pdf_text = extract_text_from_pdf(pdf_path)

    oai_embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    embedding = oai_embeddings.embed_query(pdf_text)
    index.upsert(items=[(pdf_hash, embedding)])

def extract_information_from_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)

    prompt = f"If and only if it is a job posting extract title, description, location, company, posted_at(Y-m-d H:i:s), application_deadline(YYYY-MM-DD), disability_category, number of posts as no_posts and salary rannge as salary_min, salary_max from the following PDF text:\n\n{pdf_text}"
    response = openai.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "return a json output",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    print(response)
    return response.choices[0].message.content

def fetch_jobs():
    url = base + "/recruitment.php"
    pdf_links = get_pdf_links(url)

    for pdf_link in pdf_links:
        pdf_path = f"downloaded_pdfs/{pdf_link.split('/')[-1]}"
        download_pdf(f"{base}/{pdf_link}", pdf_path)
        sleep(10)
        json_string = extract_information_from_pdf(pdf_path)
        
        data = json.loads(json_string)
        print(data)
        try:
            Job(**data).save()
        except Exception as e:
            print(e)

