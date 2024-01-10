import imaplib
import email
import os
import openai
import PyPDF2
import tempfile
import logging
import re
from celery_config import celery
from celery import group, chain
from datetime import datetime, timedelta
from email.header import decode_header
from google.cloud import secretmanager, storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set up OpenAI API Key
def access_secret_version(project_id, secret_id, version_id):
    try:
        logging.info("Trying to access secret version...")
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        logging.info("Successfully accessed secret version.")
        return response.payload.data.decode('UTF-8')
    except Exception as e:
        logging.error(f"Error accessing secret: {e}")
        return None
try:
    openai.api_key = access_secret_version('journo-396520', 'OPENAI_API_KEY', 'latest')
except Exception as e:
    logging.error(f"Error accessing OpenAI API Key: {e}")


def summary_exists(date):
    exists = os.path.exists(f'summary_{date}.txt')
    logging.info(f"Summary file for {date} exists: {exists}")
    return exists

@celery.task
def save_summary(date, summary):
    try:
        with open(f'summary_{date}.txt', 'w') as f:
            f.write(summary)
    except Exception as e:
        logging.error(f"Error saving summary for {date}: {e}") 

def load_summary(date):
    try:
        with open(f'summary_{date}.txt', 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error saving summary for {date}: {e}")
        return ""

def login_to_mailbox(email_address, password):
    IMAP_URL = 'imap.gmail.com'
    try:
        mail = imaplib.IMAP4_SSL(IMAP_URL)
        mail.login(email_address, password)
        logging.info(f"Logged in with IMAP capabilities: {mail.capabilities}")
        return mail
    except Exception as e:
        logging.error(f"Error during login: {e}")
        return None
def remove_text_before_keywords(text, keywords):
    index = text.find(keywords)
    if index != -1:
        return text[index:]
    else:
        return text  # or return None or "", depending on what you prefer

def search_email(mail, sender_email, date_to_search):
    try:
        mail.select('inbox')
        status, email_ids = mail.search(None, f'(FROM "{sender_email}" ON "{date_to_search}")')
        email_ids = email_ids[0].split()
        logging.info(f"Found {len(email_ids)} emails from {sender_email} on {date_to_search}")
        return email_ids
    except Exception as e:
        logging.error(f"Error during email search: {e}")
        return []
def get_attachment_filename_from_raw_email(raw_email):
    # Use regular expression to find the filename pattern
    match = re.search(r'Content-Disposition: attachment; filename="(.+?)"', raw_email)
    
    if match:
        # Extract the filename
        filename = match.group(1)
        return filename
    else:
        return None


def download_attachment(mail, email_id):
    allowed_filenames = ["DailyArrest_Media.pdf", "DailyCitationMedia1.pdf", "DailyReports_Media.pdf"]
    temp_dir = tempfile.mkdtemp()
    filepath = None
    try:
        _, response = mail.fetch(email_id, '(RFC822)')

        raw_email_string = response[0][1].decode('utf-8')  # Decode from bytes to string
        fileName = get_attachment_filename_from_raw_email(raw_email_string) #get file name
        if fileName not in allowed_filenames:
            logging.warning(f"Skipping unrecognized file: {fileName}")
            return None, None
        m_email = email.message_from_bytes(response[0][1])
        for part in m_email.walk():
            content_disposition = part.get("Content-Disposition", "")
            if "attachment" in content_disposition.lower():
                filepath = os.path.join(temp_dir, fileName)
                payload = part.get_payload(decode=True)
                with open(filepath, 'wb') as f:
                    f.write(payload)
                break
        return filepath, fileName
    except Exception as e:
        logging.error(f"Error during attachment download: {e}")
        return None
    

def parse_pdf_content(filepath):
    content = []
    try:
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                content.append(page.extract_text())
        extracted_length = len('\n'.join(content))
        logging.info(f"Extracted {extracted_length} characters from PDF {filepath}")
    except Exception as e:
        logging.error(f"Error parsing PDF content: {e}")
        return ""
    return '\n'.join(content)

def get_file_from_gcs(bucket_name, blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text()
    except Exception as e:
        logging.error(f"Error fetching file from GCS: {e}")
        return ""

@celery.task
def summarize_with_openai(content, filename):
    try:
        if filename == "DailyArrest_Media.pdf":
            prompt_file_name = 'emailRestyleArrests.txt'
            content = remove_text_before_keywords(content,"Media Arrest Report")
        elif filename == "DailyCitationMedia1.pdf":
            prompt_file_name = 'emailRestyleCitations.txt'
            content = remove_text_before_keywords(content,"Media Citation Report")
        elif filename == "DailyReports_Media.pdf":
            prompt_file_name = 'emailRestyleIncidents.txt'
            content = remove_text_before_keywords(content,"Media Incident Report")
        else:
            logging.warning(f"Unrecognized filename: {filename}. Using default prompt.")
            prompt_file_name = 'emailRestyleArrests.txt'

        promptFile = get_file_from_gcs("journo-text-data", prompt_file_name)
        
        full_content = promptFile + content

        messages = [
            {"role": "system", "content": "You are an assistant that reformats text into bulleted lists"},
            {"role": "user", "content": full_content}
        ]
        logging.info(f"Content being sent to OpenAI: {full_content}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
            top_p=1.0
        )

        summarized_text = response['choices'][0]['message']['content']
        logging.info(f"Received response from OpenAI: {summarized_text}")
        return summarized_text
    except Exception as e:
        logging.error(f"Error summarizing with OpenAI: {e}")
        return "" 
@celery.task
def concatenate_summaries(date, content):
    returnString = ""
    returnString = returnString + str(date) + "/n"
    for summary in content:
        returnString += summary + "/n"
    return returnString

def main():
    try:
        EMAIL = access_secret_version('journo-396520', 'EMAIL', 'latest')
        PASSWORD = access_secret_version('journo-396520', 'PASSWORDUPDATED', 'latest')
    except Exception as e:
        logging.error(f"Error fetching secrets: {e}")
        return []
    SENDER_EMAIL = 'AWintermote@rileycountypolice.org'
    task_ids = []

    today = datetime.now()
    # Calculate dates for the last 5 days
    dates_to_search = [(today - timedelta(days=i)).strftime('%d-%b-%Y') for i in range(4)]

    all_summaries = {}
    mail = login_to_mailbox(EMAIL, PASSWORD)
    for date_to_search in dates_to_search:
        daily_tasks = []
        summary = ""
        try:
            if summary_exists(date_to_search):
                # If summary already exists, load it from the file
                summary = load_summary(date_to_search)
            else:
                email_ids = search_email(mail, SENDER_EMAIL, date_to_search)
                filePaths = []
                daily_summaries = []
                for email_id in email_ids:
                    attachment, filename = download_attachment(mail, email_id)
                    if attachment is None and filename is None:
                        continue
                    filePaths.append(attachment)
                    content = parse_pdf_content(attachment)
                    daily_tasks.append(summarize_with_openai.s(args=[content, filename]))
                    # Concatenate all summaries of the day
            summarization_group = group(daily_summaries)
            pipeline = chain(
                    group(*daily_tasks),
                    concatenate_summaries.s(date_to_search),
                    save_summary.s(date_to_search)
                )

            result = pipeline.apply_async()
            task_ids.append(result.id)
                
        except Exception as e:
            logging.error(f"Error processing data for {date_to_search}: {e}")
        all_summaries[date_to_search] = summary
    return {'task_ids': task_ids, 'all_summaries': all_summaries}

if __name__ == "__main__":
    main()
