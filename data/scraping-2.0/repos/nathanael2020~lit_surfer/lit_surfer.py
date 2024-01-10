#lit_surfer.py
import json
import requests
import xml.etree.ElementTree as ET
import datetime
import json
import os
import subprocess
import shutil
import smtplib
import argparse
import PyPDF2

from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

together_api = os.getenv('TOGETHER_API_KEY')
google_password = os.getenv('GOOGLE_PASSWORD')
receiver_email = os.getenv('RECEIVER_EMAIL')
sender_email = os.getenv('SENDER_EMAIL')
file_prefix = os.getenv('FILE_PREFIX') #usage /path/to/lit_surfer

print(sender_email)

# Define the downloaded_pdfs directory
downloaded_pdfs_dir = f"{file_prefix}/downloaded_pdfs/"
language_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#language_model = "gpt-4-1106-preview"
api_base_url = "https://api.together.xyz"
#api_base_url = "https://api.openai.com/v1"

# Ensure the downloaded_pdfs directory exists
if not os.path.exists(downloaded_pdfs_dir):
    os.makedirs(downloaded_pdfs_dir)

# $OPEN_API_KEY
client = OpenAI(
    api_key=f'{together_api}',
    base_url=api_base_url,
)

missing_papers_list = []

def download_pdfs(papers, download_folder=f"{file_prefix}/downloads"):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for paper in papers:
        pdf_url = paper["pdf_url"]
        pdf_filename = pdf_url.split('/')[-1]
        if not pdf_filename.lower().endswith('.pdf'):
            pdf_filename += '.pdf'

        # Check if the PDF already exists in downloaded_pdfs
        downloaded_pdf_path = os.path.join(downloaded_pdfs_dir, pdf_filename)
        if os.path.exists(downloaded_pdf_path):
            print(f"{pdf_filename} already downloaded in downloaded_pdfs.")
            # Copy the PDF to downloads instead of downloading it again
            destination_path = os.path.join(download_folder, pdf_filename)
            shutil.copy(downloaded_pdf_path, destination_path)
            print(f"Copied {pdf_filename} to downloads.")
            continue

        # Download the PDF if it doesn't exist in downloaded_pdfs
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_path = os.path.join(download_folder, pdf_filename)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
                print(f"Downloaded {pdf_path}")
        else:
            print(f"Failed to download {pdf_url}")
            missing_papers_list.append(pdf_filename)

# Add a function to move PDFs from downloads to downloaded_pdfs
def move_pdf_to_downloaded_pdfs(pdf_file):

    if os.path.exists(os.path.join(f"{file_prefix}downloads", pdf_file)):
        src = os.path.join(f"{file_prefix}downloads", pdf_file)
        dst = os.path.join(downloaded_pdfs_dir, pdf_file)
        shutil.move(src, dst)
        print(f"Moved {pdf_file} to downloaded_pdfs")

def query_arxiv(search_term, start_num, max_results):

    def make_request(url):
        print(url)
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Error in API request")
        return response
    print(max_results)
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=all:{search_term}&sortBy=submittedDate&sortOrder=descending&start={start_num}&max_results={max_results}"
    cleaned_query = query.replace('"', '').replace("''", "'")
    
    response = make_request(base_url + cleaned_query)

    root = ET.fromstring(response.content)
    total_results = int(root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text)

    if total_results == 0:
        # If no results, switch to HTTP
        base_url = "https://export.arxiv.org/api/query?"
        response = make_request(base_url + cleaned_query)
        root = ET.fromstring(response.content)
        # Recheck total results with HTTP request
        total_results = int(root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text)
        if total_results == 0:
            print("No results found for the query on both HTTPS and HTTP.")
            return []
        
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
        published_date = entry.find('{http://www.w3.org/2005/Atom}published').text
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        # Extract PDF URL
        pdf_url = entry.find('{http://www.w3.org/2005/Atom}link[@title="pdf"]').attrib['href']
        papers.append({
            "title": title.strip(),
            "abstract": abstract.strip(),
            "published_date": published_date.strip(),
            "authors": authors,
            "pdf_url": pdf_url,
            "scraped_time": datetime.datetime.now().isoformat(),
        })

    return papers

def create_summarizer_prompt(memory):
    return f"{memory}\n\nSummarize this material in no more than six " \
        "paragraphs, first two paragraphs summarizing the research and " \
        "results, then two more paragraphs describe how this research fits " \
        "into the existing body of research. Then include two more " \
        "paragraphs critiquing the research. Before responding, rewrite " \
        "it at least 3 times, improving it each time. Think deeply on the " \
        "research context, the achievements of this research, what the " \
        "authors peers would find lacking, and what a skeptical critic " \
         "would admit is valuable. Respond with the best final draft of " \
        "your six paragraph summary. Preface your response with " \
        "'Summary:\n\n'"

def create_summary_critic_prompt(memory):
    return f"{memory}\n\nReview the original paper and evaluate the quality " \
    "of the summary. Identify 5 points of weakness that need improvement. " \
    "Respond with details on how to improve the summary. Don't resummarize " \
    "yourself. Preface your response with 'Critique:\n\n'"

def create_improved_summary_prompt(memory):
    return f"{memory}\n\nReview the original paper, the first draft summary, and the points of needed improvement. Rewrite the summary in exactly six paragraphs considering these points. Then rewrite it again 4 times, improving it each time, and really considering the paper with a skeptical eye. Be sure to check all your facts by thoroughly reviewing the paper. Respond only with your best final draft of the six paragraphs, not the first three revisions."

def create_final_summary_prompt(memory):
    return f"{memory}\nReview the paper and the draft summary and refine the summary further, condense it to exactly four very well-written paragraphs. Be sure to check all your facts by thoroughly reviewing the paper and rewriting a final time. Respond only with your four-paragraph final summary."

def create_final_tldr_prompt(memory):
    return f"{memory}\nReview the paper and the draft summary and refine the summary further. Be sure to check all your facts by thoroughly reviewing the paper. Reply with a three sentence 'tl;dr' summary explaining where this fits into the broader field of research and why it's important, as if you're talking to a general audience. No more than three sentences, and no more than one paragraph."

def call_gpt_summarizer(prompt):
    """Makes a call to the GPT-4 API and returns the response."""
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=language_model,
        max_tokens=2000
    )

    return response.choices[0].message.content

def append_to_json_file(filename, content):
    try:
        # Read existing data from the file
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        data = []

    # Append the new content
    data.append(content)

    # Write the updated data back to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
        
def read_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def extract_content(paper_content):

    # Extract all the content subsections from within the body section
    # Only keep the first 2,000 characters of the paper and the last 2,000 characters before the References section.
    #
    # This is to respect token limits and also to economize the cost of processing, which is paid per token.
    # An improvement might be to segment long papers and recursively summarize sub-sections and then sections, and then the paper as a whole.

    paper_text = ' '.join(' '.join(section['content']) for section in paper_content.get('body', []) if 'content' in section)

    # Extract the first 1000 characters
    first_2000_chars = paper_text[:2000]

    # Find the start of the references section
    references_index = paper_text.find("References")
    if references_index != -1:
        start_index = max(0, references_index - 2000)
        before_references = paper_text[start_index:references_index]
    else:
        before_references = paper_text[-2000:]

    return first_2000_chars, before_references
 
def list_pdf_filenames(directory=f'{file_prefix}/downloads'):
    pdf_files = [file for file in os.listdir(directory) if file.endswith('.pdf')]
    return pdf_files

def reformat_json_content(json_data):
    formatted_data = []

    for entry in json_data:
        # Split the text into paragraphs at each occurrence of "\n\n"
        paragraphs = entry.split("\n\n")
        
        # Add the paragraphs to the formatted data
        formatted_data.append({"section": paragraphs})

    return formatted_data

def process_and_save_api_response(response, processed_data_filepath, string_arg, int_arg):
    processed_data = {}

    for item in response:
        # Extracting required fields
        key = item['pdf_url'].split('/')[-1]  # Removing '.pdf' from the url
        processed_data[key] = {
            'title': item.get('title', ''),
            'authors': item.get('authors', []),
            'abstract': item.get('abstract', ''),
            'published_date': item.get('published_date', ''),
            'scraped_time': item.get('scraped_time', ''),
            'pdf_url': item.get('pdf_url', '')
        }

    # Save to a temporary JSON file
    with open(processed_data_filepath, 'w') as file:
        json.dump(processed_data, file, indent=4)

    return "Data processed and saved to 'processed_api_data.json'"

# Function to reformat the date or return a default message on error
def format_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').strftime('%m/%d/%Y')
    except (ValueError, TypeError):
        return "No Valid Date"

def extract_with_pypdf(pdf_path, output_json_filepath):

    content = ""

    try:
        # Try to open and read the PDF file
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in range(len(reader.pages)):
                content += reader.pages[page].extract_text()

    except (PyPDF2.errors.PdfReadError, OSError) as e:
        # Handle corrupted or unreadable file
        print(f"Error with {pdf_path}: {e}. Deleting file and adding to missing papers.")
        
        # Delete the corrupted file
        os.remove(pdf_path)

        filename = pdf_path.split('/')[-1]
        
        # Remove the file extension
        paper_id = '.'.join(filename.split('.')[:-1])

        # Add to the missing papers list
        missing_papers_list.append(paper_id)

    output_dir = os.path.dirname(output_json_filepath)
    print(output_dir)

    """
    Writes the extracted content to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(output_json_filepath)
    with open(output_json_filepath, 'w') as json_file:
        json.dump({"body": content}, json_file)

def generate_requirements_file():
    # This function will run 'pip freeze' and write its output to 'requirements.txt'
    try:
        with open('requirements.txt', 'w') as file:
            subprocess.run(['pip', 'freeze'], stdout=file, check=True)
        print("requirements.txt file generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while generating requirements.txt: {e}")

def main(string_arg, start_num_arg, int_arg):
    # Your code here
    print(f"Received string argument: {string_arg}")
    print(f"Received integer argument: {int_arg}")

    search_term = f"'{string_arg}'"

    papers = query_arxiv(search_term, start_num=start_num_arg, max_results=int_arg)

    print(papers)

    cleaned_search_term = search_term.replace("'","").replace('"','')                                                      

    processed_data_filepath = f"{file_prefix}/{cleaned_search_term}_{int_arg}_processed_api_data.json"

    process_and_save_api_response(papers, processed_data_filepath, string_arg, int_arg)

    download_pdfs(papers)

    pdf_filenames = list_pdf_filenames()

    for pdf_fn in pdf_filenames:
        print(pdf_fn)

    for paper in papers:

        file = paper['pdf_url'].split('/')[-1]  # Removing '.pdf' from the url
        
        processed_data = read_json_file(processed_data_filepath)

        json_filename = f"{file_prefix}/{file}/{file}.json"

        print(json_filename)

        pdf_filepath = f"{file_prefix}/downloads/{file}.pdf"
        pdf_file = f"{file}.pdf"

        # Try with appjsonify
#        command = ['/home/nmiksis_gmail_com/.venv/bin/appjsonify', pdf_filepath, '/home/nmiksis_gmail_com/jsonapp/jsonhandler/', '--paper_type', f'{paper_arg}']
        # if not os.path.exists(json_filename):
        #     subprocess.run(command)

        # Try with PyPDF
        if os.path.exists(pdf_filepath):
            if not os.path.exists(json_filename):
                print(json_filename, " ", pdf_filepath)
                extract_with_pypdf(pdf_filepath, json_filename)

            # After processing the papers
        move_pdf_to_downloaded_pdfs(pdf_file)

        if os.path.exists(pdf_filepath):
            os.remove(pdf_filepath)

        # Check if the JSON file exists
        if not os.path.exists(json_filename):
            print(f"JSON file for {file}.pdf does not exist. Skipping...")

            missing_papers_list.append(file)

            continue

        json_content = read_json_file(f"{file_prefix}/{file}/{file}.json")

        if os.path.exists(json_filename):
            os.remove(json_filename)

        if os.path.exists(json_filename):
            os.remove(json_filename)

            directory_path = f"{file_prefix}/{file}/"

            # Check if the directory exists and then remove it
            if os.path.isdir(directory_path):
                shutil.rmtree(directory_path)
                
        abstract = next((paper['abstract'] for paper in papers if paper['pdf_url'].endswith(f'{file}')), "Abstract not found.")

        prompt = create_summarizer_prompt(f"{extract_content(json_content)}, {abstract}")
        first_summary_response = call_gpt_summarizer(prompt)
        prompt = create_summary_critic_prompt(f"{extract_content(json_content)}, {abstract}, {first_summary_response}")
        critic_response = call_gpt_summarizer(prompt)
        prompt = create_improved_summary_prompt(f"{extract_content(json_content)}, {abstract}, {first_summary_response}, {critic_response}")
        response = call_gpt_summarizer(prompt)
        prompt = create_final_summary_prompt(f"{extract_content(json_content)}, {response}")
        response = call_gpt_summarizer(prompt)

        prompt = create_final_tldr_prompt(f"{extract_content(json_content)}, {response}")
        tldr = call_gpt_summarizer(prompt)

        # Check if 'foo' key exists in processed_data
        if file in processed_data:
            # Append the 'body' content from foo.json
            processed_data[file]['body'] = json_content['body']
            processed_data[file]['gpt_response'] = response
            processed_data[file]['tldr'] = tldr
            
        else:
            print("'foo' key not found in processed_data")

        with open(processed_data_filepath, 'w') as file:
            json.dump(processed_data, file, indent=4)

    # remove missing papers from processed_data
            
    for missing_paper in missing_papers_list:

        print(f"Missing paper: {missing_paper}")

        processed_data = read_json_file(processed_data_filepath)

        # Check if the entry exists and delete it
        entry_to_delete = missing_paper
        if entry_to_delete in processed_data:
            print(f"Removing missing paper: {missing_paper}")
            del processed_data[entry_to_delete]
            if entry_to_delete in papers:
                papers.remove(entry_to_delete)

        # Write the updated data back to the file
        with open(processed_data_filepath, 'w') as file:
            json.dump(processed_data, file, indent=4)

    if google_password:
        email_body = ""

        # write tldrs to email body

        processed_data = read_json_file(processed_data_filepath)
        print(processed_data)

        papers = processed_data
        print(papers)
        email_body += "********************** TL;DRs *********************\n"

        for paper in processed_data:

            formatted_date = datetime.datetime.fromisoformat(processed_data[paper]['published_date'].rstrip("Z")).strftime("%m/%d/%Y")

            email_body += f"""
            Title: {processed_data[paper]['title']} ({formatted_date})\n
            Tl;dr: {processed_data[paper]['tldr']}\n
            """
        
        email_body += "***************** Longer Summaries ****************\n"

        for paper in processed_data:
            
            formatted_date = datetime.datetime.fromisoformat(processed_data[paper]['published_date'].rstrip("Z")).strftime("%m/%d/%Y")

            email_body += f"""
            Title: {processed_data[paper]['title']}\n
            Authors: {processed_data[paper]['authors']}\n
            Published Date: {formatted_date}\n
            PDF URL: {processed_data[paper]['pdf_url']}\n
            GPT Summary: {processed_data[paper]['gpt_response']}\n
            **************************************************\n\n
            """

        smtp_server = "smtp.gmail.com"
        smtp_port = 587  # or 25, or 465 (for SSL)
        smtp_user = sender_email
        smtp_password = google_password
        
        subject = f"Arxiv GPT Summaries: {search_term} (papers {start_num_arg + 1} to {start_num_arg + int_arg})"

        if papers == {}:

            email_body = f"No results returned for {search_term}"

        text_content = email_body
        html_content = text_content.replace("\n", "<br>")

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = receiver_email

        # Attach both plain text and HTML versions
        message.attach(MIMEText(html_content, "html"))

        if papers != {}:
            # Send the email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_email, receiver_email, message.as_string())

            print("Email sent successfully!")

        else:
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_email, receiver_email, message.as_string())

            print("Email sent, but no papers returned.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python script with three arguments.')
    parser.add_argument('string_arg', type=str, help='A string argument')
    parser.add_argument('start_num_arg', type=int, help='A start num integer argument')
    parser.add_argument('int_arg', type=int, help='An integer argument')

    args = parser.parse_args()
    main(args.string_arg, args.start_num_arg, args.int_arg)