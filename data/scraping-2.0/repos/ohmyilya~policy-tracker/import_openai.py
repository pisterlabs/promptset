import os
import openai
import requests
from bs4 import BeautifulSoup
import csv
from PyPDF2 import PdfReader  # Use PdfReader from PyPDF2
import tempfile

# Set your OpenAI API key here
api_key = "Open AI key"
openai.api_key = api_key

# Function to download PDFs from a webpage and save their URLs to a CSV file
def download_pdfs_and_save_urls(url, csv_file):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    pdf_links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.pdf'):
            pdf_links.append(href)  # Remove the base URL prefix
    
    # Save the PDF URLs to a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ["PDF URL"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pdf_url in pdf_links:
            writer.writerow({"PDF URL": pdf_url})

# Function to extract text from a PDF given its URL
def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
        temp_pdf_file.write(response.content)
    
    text = ""
    with open(temp_pdf_file.name, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    
    os.remove(temp_pdf_file.name)  # Clean up the temporary PDF file
    return text

# Define the webpage URL for PDF downloads
webpage_url = "https://www.asuris.com/provider/library/bulletins"
pdf_urls_csv = "pdfs_urls.csv"

# Download PDFs and save their URLs to the CSV file
download_pdfs_and_save_urls(webpage_url, pdf_urls_csv)

# Function to summarize text using the OpenAI API
def summarize_text(api_key, input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Define the CSV file for data storage
data_csv_file = "data_collection.csv"

# Read the PDF URLs from the CSV file and extract text from each PDF
with open(pdf_urls_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pdf_url = row["PDF URL"]
        pdf_text = extract_text_from_pdf(pdf_url)
        
        # You can add code here to extract the required information (Date of Notice, Health Plan, etc.)
        # from the PDF text based on your specific format
        
        # Summarize the extracted text using the OpenAI API
        summary = summarize_text(api_key, pdf_text)
        
        # Append the summary to the data CSV file
        with open(data_csv_file, 'a', newline='') as data_csv:
            fieldnames = ["Date of Notice", "Health Plan", "Line of Business", "Effective Date", 
                          "Summary of Update", "Link to Update", "Topic", "Impacted Departments", 
                          "Dispute with Payer? (Y/N)", "Staff Comments"]
            writer = csv.DictWriter(data_csv, fieldnames=fieldnames)
            
            # You should populate the row with the extracted information and summary
            writer.writerow({"Date of Notice": "TODO",
                             "Health Plan": "TODO",
                             "Line of Business": "TODO",
                             "Effective Date": "TODO",
                             "Summary of Update": summary,
                             "Link to Update": pdf_url,
                             "Topic": "TODO",
                             "Impacted Departments": "TODO",
                             "Dispute with Payer? (Y/N)": "TODO",
                             "Staff Comments": "TODO"})

# You can add further processing or summarization as required
