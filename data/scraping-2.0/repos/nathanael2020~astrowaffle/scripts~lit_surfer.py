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

from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv())
from openai import OpenAI
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

email_logo_url = f"{os.getenv('STATIC_IMAGES_DIR_URL')}/dalle_astrowaffle.png"
accounts_home_url = os.getenv('ACCOUNTS_HOME_URL')
api_key = os.getenv('TOGETHER_API_KEY')
#api_key = os.getenv('OPEN_AI_API_KEY')
google_password = os.getenv('GOOGLE_PASSWORD')
#receiver_email = os.getenv('RECEIVER_EMAIL')
sender_email = os.getenv('SENDER_EMAIL')
file_prefix = os.getenv('FILE_PREFIX') #usage /path/to/lit_surfer

# Define the downloaded_pdfs directory
downloaded_pdfs_dir = f"{file_prefix}/downloaded_pdfs"
language_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#language_model = "gpt-4-1106-preview"
api_base_url = "https://api.together.xyz"
#api_base_url = "https://api.openai.com/v1"

# Ensure the downloaded_pdfs directory exists
if not os.path.exists(downloaded_pdfs_dir):
    os.makedirs(downloaded_pdfs_dir)

# $OPEN_API_KEY
client = OpenAI(
    api_key=f'{api_key}',
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

    if os.path.exists(os.path.join(f"{file_prefix}/downloads", pdf_file)):
        src = os.path.join(f"{file_prefix}/downloads", pdf_file)
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
    return f"{memory}\n\nReview the original paper, the first draft summary, " \
        "and the points of needed improvement. Rewrite the summary in exactly " \
        "six paragraphs considering these points. Then rewrite it again 4 times," \
        " improving it each time, and really considering the paper with a skeptical " \
        "eye. Be sure to check all your facts by thoroughly reviewing the paper. " \
        "Respond only with your best final draft of the six paragraphs, not the first " \
        "three revisions."

def create_final_summary_prompt(memory):
    return f"{memory}\nReview the paper and the draft summary and refine the summary " \
        "further, condense it to exactly four very well-written paragraphs. Be sure to " \
        "check all your facts by thoroughly reviewing the paper and rewriting a final " \
        "time. Respond only with your four-paragraph final summary."

def create_final_tldr_prompt(memory):
    return f"{memory}\nReview the paper and the draft summary and refine the summary " \
    "further. Be sure to check all your facts by thoroughly reviewing the paper. Reply" \
    " with a three sentence 'tl;dr' summary explaining where this fits into the broader " \
    "field of research and why it's important, as if you're talking to a general audience." \
    "No more than three sentences, and no more than one paragraph."

def create_topic_keyword_prompt(memory):
    return f"{memory}\nReview the abstract. Reply with five keywords that best identify " \
    "the topic covered in the abstract. Common keywords are 'machine learning', 'quantum " \
    "physics', 'botany', 'health', 'medicine', etc. Try to make keywords descriptive but " \
    "high-level enough to indicate the general field of research as well as the sub-specialty. " \
    "Do not number your keywords. Respond with your keywords bounded with single quotes, " \
    "separated by a comma between them."

def create_like_im_fifteen_prompt(memory):
    return f"{memory}\nReview the original paper, the abstract, the keywords, and the summary. " \
    "This is very important. I need you to start with the paper and the abstract and rewrite " \
    "the summary for a smart fifteen-year-old. I need you to work really hard to borrow from " \
    "everything you know on the topic to write an excellent summary for this very important " \
    "fifteen-year-old person who loves to learn. Review the material several times to make sure" \
    " you understand it. Rewrite your response 4 times, improving it each time. Make it engaging, " \
    "understandable, exciting, and accurate, both with respect to the paper and to your knowledge" \
    " on the topic. This is important that you do a good job! Reply only with your 3 paragraphs summary."

def create_like_im_a_freshman_prompt(memory):
    return f"{memory}\nReview the original paper, the abstract, the keywords, and the summary. " \
    "This is very important. I need you to start with the paper and the abstract and rewrite the " \
    "summary for a smart, ambitious college freshman. I need you to work really hard to borrow from " \
    "everything you know on the topic to write an excellent summary for this very important young " \
    "person who loves to learn. Review the material several times to make sure you understand it." \
    " Rewrite your response 4 times, improving it each time. Make it engaging, understandable, " \
    "exciting, and accurate, both with respect to the paper and to your knowledge on the topic. " \
    "This is important that you do a good job! Reply only with your 3 paragraph summary."

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

def create_button(keyword, index):
    cleaned_keyword = keyword.replace("'", "").upper()

    colors = [
        "#007bff",  # Blue
        "#28a745",  # Green
        "#dc3545",  # Red
        "#ffc107",  # Yellow
        "#17a2b8"   # Cyan
    ]
    color = colors[index % len(colors)]
    return f'''<a style="background-color: {color}; color: white; padding: 10px 20px; text-align: center; 
            text-decoration: none; display: inline-block; border-radius: 5px; margin: 2px;" href="#">{cleaned_keyword}</a>
            '''


def main(string_arg, start_num_arg, int_arg, email_arg, like_im_fifteen_arg, like_im_a_freshman_arg):
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

    print(f"like_im_fifteen_arg: {str(like_im_fifteen_arg)}")
    print(f"like_im_a_freshman_arg: {str(like_im_a_freshman_arg)}")

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
        improved_response = call_gpt_summarizer(prompt)
        prompt = create_final_summary_prompt(f"{extract_content(json_content)}, {improved_response}")
        response = call_gpt_summarizer(prompt)

        prompt = create_final_tldr_prompt(f"{extract_content(json_content)}, {response}")
        tldr = call_gpt_summarizer(prompt)

        prompt = create_topic_keyword_prompt(f"{abstract}")
        keywords = call_gpt_summarizer(prompt)

        like_im_fifteen_response = ""

        if(like_im_fifteen_arg == "True"):
            prompt = create_like_im_fifteen_prompt(f"{extract_content(json_content)}, {abstract}, {improved_response}, {keywords}")
            like_im_fifteen_response = call_gpt_summarizer(prompt)

        like_im_a_freshman_response = ""

        if(like_im_a_freshman_arg == "True"):
            prompt = create_like_im_a_freshman_prompt(f"{extract_content(json_content)}, {abstract}, {improved_response}, {keywords}")
            like_im_a_freshman_response = call_gpt_summarizer(prompt)

        # Check if 'foo' key exists in processed_data
        if file in processed_data:
            # Append the 'body' content from foo.json
            processed_data[file]['body'] = json_content['body']
            processed_data[file]['gpt_response'] = response
            processed_data[file]['tldr'] = tldr
            processed_data[file]['keywords'] = keywords
            processed_data[file]['like_im_fifteen'] = like_im_fifteen_response
            processed_data[file]['like_im_a_freshman'] = like_im_a_freshman_response
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
#        print(papers)

        email_body += f"""
        
        <html>
            <head>
                <link href="https://fonts.googleapis.com/css?family=Montserrat"
                rel="stylesheet">
            </head>
                <body>
                    <div style="width: 100%; text-align: center">
                        <a href="{accounts_home_url}">
                            <img width="180px" src="{email_logo_url}">
                        </a>
                    </div>
                    <table style="background: #007bff; width: 100%; margin: 0px; padding-bottom: 20px">
                        <tr>
                            <td style="font-family: 'Montserrat', sans-sarif; margin: 0px; padding: 0px 0px;">
                                <div style="text-align: center; margin: 30px 0px 0px 0px; font-size: 20pt; color: #fff">
                                <h2>TL;DRs</h2></div>
                            </td>

                        </tr>
                    <tr>
                        <td style="font-family: 'Montserrat'; border-left: 0px solid #f59; padding: 0px 20px; margin: 0px">
                        <div style="background: #fff; margin: 10px; padding:10px 30px">"""

        for paper in processed_data:

            # formatted_date = datetime.datetime.fromisoformat(processed_data[paper]['published_date'].rstrip("Z")).strftime("%m/%d/%Y")
            keywords = processed_data[paper]['keywords'].split(', ')
            keywords_buttons = ' '.join([create_button(kw, i) for i, kw in enumerate(keywords)])

            cleaned_title = processed_data[paper]['title'].replace("\n", "")

            email_body += f'''
                        <div class="container" style="background: #fff; width: 100%; margin: 0px auto">
                            <p style="background: #ffbf66; color: black; font-size: 18pt; font-weight: 700; 
                                text-align: top; padding: 10px">{cleaned_title}</h2><br>
                            </p>
                        </div>
                        <div class="container" style="width: 100%; margin: 0 auto">
                            <div class="solid-bar" style="border-left: 0px solid #007bff; background-color: #fff; padding: 0px 0px; margin: 10px 0px;">
                                <p style="margin-bottom: 10px; color: black; font-size: 11pt; padding: 0px 0px">
                                </p>
                            </div>
                            <div class="container" style="width: 100%; margin: 0 auto">
                                <div class="solid-bar" style="border-left: 0px solid #007bff; background-color: #fff; padding: 0px 0px; margin: 10px 0px;">
                                    <p style="margin-bottom: 10px; color: black; font-size: 11pt; padding: 0px 0px">
                                        <span>{keywords_buttons}</span><br>
                                    </p>
                            </div>
                            <div class="solid-bar" style="border-left: 10px solid #000; background-color: #fff; 
                                    padding: 0px 20px; margin: 10px 0px 60px 0px;">
                                <p style="margin-bottom: 0px; vertical-align: top; top: 15px; color: black; 
                                    font-size: 12pt; padding: 10px 0px">AI Summary: {processed_data[paper]['tldr']}
                                </p>
                            </div>
                '''
        
            cleaned_like_im_fifteen_summary = processed_data[paper]['like_im_fifteen'].replace("\n", "<br>")
            cleaned_like_im_a_freshman_summary = processed_data[paper]['like_im_a_freshman'].replace("\n", "<br>")

            print(f"like_im_fifteen_arg: {str(like_im_fifteen_arg)}")
            print(f"like_im_a_freshman_arg: {str(like_im_a_freshman_arg)}")
            
            if(like_im_fifteen_arg == "True"):
                email_body += f'''
                        <div class="solid-bar" style="border-left: 10px solid #000; background-color: #28a745; 
                                padding: 0px 20px; margin: 10px 0px 60px 0px;">
                            <p style="margin-bottom: 0px; vertical-align: top; top: 15px; color: #000; 
                                font-size: 12pt; padding: 10px 0px">AI "Like I'm Fifteen" Summary: {cleaned_like_im_fifteen_summary}
                            </p>
                        </div>
                        '''

            if(like_im_a_freshman_arg == "True"):
                email_body += f'''
                        <div class="solid-bar" style="border-left: 10px solid #000; background-color: 
                                #17a2b8; padding: 0px 20px; margin: 10px 0px 60px 0px;">
                            <p style="margin-bottom: 0px; vertical-align: top; top: 15px; color: #000; font-size: 
                                12pt; padding: 10px 0px">AI "Like I'm a College Freshman" Summary: {cleaned_like_im_a_freshman_summary}
                            </p>
                        </div>
                        '''

        email_body += f'''
                    </td>
                </tr>
            </table>    
            <table style="background: #ffc107; width: 100%; margin: 60px 0px; padding-bottom: 20px">            
                <tr>
                    <td style="font-family: 'Montserrat', sans-sarif; margin: 0px; padding: 0px 0px;">
                        <div style="text-align: center; margin: 30px 0px 0px 0px; font-size: 20pt; color: #000"><h2>Longer Summaries</h2></div>
                    </td>
                </tr>
                <tr>
                    <td style="font-family: 'Montserrat'; border-left: 0px solid #f59; padding: 0px 20px; margin: 0px">
                        <div style="background: #fff; margin: 10px; padding:10px 30px">
            '''

        for paper in processed_data:
            
            formatted_date = datetime.datetime.fromisoformat(processed_data[paper]['published_date'].rstrip("Z")).strftime("%m/%d/%Y")

            keywords = processed_data[paper]['keywords'].split(', ')
            keywords_buttons = ' '.join([create_button(kw, i) for i, kw in enumerate(keywords)])
            cleaned_title = processed_data[paper]['title'].replace("\n", "")
            cleaned_gpt_summary = processed_data[paper]['gpt_response'].replace("\n", "<br>")

            email_body += f'''
                            <div class="container" style="background: #fff; width: 100%; margin: 0px auto">
                                <p style="background: #ffbf66; color: black; font-size: 18pt; font-weight: 700; text-align: top; padding: 10px">{cleaned_title}</p>
                            </div>
                            <div class="container" style="width: 100%; margin: 0 auto">
                                <div class="solid-bar" style="border-left: 0px solid #007bff; background-color: #fff; padding: 0px 0px; margin: 10px 0px;">
                                    <p style="margin-bottom: 10px; color: black; font-size: 11pt; padding: 0px 0px">{keywords_buttons}</span></p>
                                </div>
                                <div class="solid-bar" style="border-left: 10px solid #000; background-color: #fff; padding: 0px 20px; margin: 10px 0px 0px 0px;">
                                    <p style="margin-bottom: 0px; vertical-align: top; top: 15px; color: black; font-size: 11pt; padding: 10px 0px">Authors: {', '.join(processed_data[paper]['authors'])}
                                    </p>
                                </div>
                                <div class="solid-bar" style="border-left: 10px solid #000; background-color: #fff; padding: 0px 20px; margin: 10px 0px 0px 0px;">
                                    <p style="margin-bottom: 0px; vertical-align: top; top: 15px; color: black; font-size: 11pt; 
                                            padding: 10px 0px">Published Date: {formatted_date}
                                    </p>
                                </div>
                                <div class="solid-bar" style="border-left: 10px solid #000; background-color: #fff; padding: 0px 20px; margin: 10px 0px 0px 0px;">
                                    <p style="margin-bottom: 0px; vertical-align: top; top: 15px; color: black; font-size: 11pt; padding: 10px 0px">PDF URL: <a target="_blank" href="{processed_data[paper]['pdf_url']}">{processed_data[paper]['pdf_url']}</a>
                                    </p>
                                </div>
                                <div class="solid-bar" style="border-left: 10px solid #000; background-color: #fff; padding: 0px 20px; margin: 10px 0px 0px 0px;">
                                    <p style="margin-bottom: 0px; vertical-align: top; top: 15px; color: black; font-size: 11pt; padding: 10px 0px">GPT Summary:</strong> {cleaned_gpt_summary}
                                    </p>
                                </div>
                    '''

        email_body += '''   </td>
                        </tr>
                    </table>
                </body>
            </head>
        </html>
        '''

        smtp_server = "smtp.gmail.com"
        smtp_port = 587  # or 25, or 465 (for SSL)
        smtp_user = sender_email
        smtp_password = google_password

        formatted_subject_date = datetime.datetime.today().strftime("%m/%d/%Y")
        subject = f"Special AstroWaffle Robot Delivery - {formatted_date}"        
        subject = f"Special AstroWaffle Robot Delivery - {formatted_subject_date} ({search_term.upper()})"

        if papers == {}:

            email_body = f"No results returned for {search_term}"

        email_body += f"\n\n Manage your Astro Waffle Robot subscriptions at <a href='{accounts_home_url}'>{accounts_home_url}</a>."

        html_content = email_body
        # html_content = text_content.replace("\n", "<br>")

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = email_arg

        # Attach both plain text and HTML versions
        message.attach(MIMEText(html_content, "html"))

        if papers != {}:
            # Send the email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_email, email_arg, message.as_string())

            print("Email sent successfully!")

        else:
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_email, email_arg, message.as_string())

            print("Email sent, but no papers returned.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python script with three arguments.')
    parser.add_argument('string_arg', type=str, help='A string argument')
    parser.add_argument('start_num_arg', type=str, help='A start num integer argument')
    parser.add_argument('int_arg', type=str, help='An integer argument')
    parser.add_argument('email_arg', type=str, help='A string argument')
    parser.add_argument('like_im_fifteen_arg', type=str, help='A boolean argument')
    parser.add_argument('like_im_a_freshman_arg', type=str, help='A boolean argument')

    args = parser.parse_args()
    main(args.string_arg, args.start_num_arg, args.int_arg, args.email_arg, args.like_im_fifteen_arg, args.like_im_a_freshman_arg)