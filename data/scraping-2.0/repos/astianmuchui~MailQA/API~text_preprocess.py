from langchain.text_splitter import CharacterTextSplitter
from gmail_fetch import GmailAPI
import re
from bs4 import BeautifulSoup

class TextProcessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')

    def remove_html_tags(self, text):
        """Removes all HTML tags from the text"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    def remove_links(self, text):
        """Replaces URLs with an empty string"""
        text_without_links = self.url_pattern.sub('', text)
        return text_without_links

    def preprocess_email_data(self, email_data):
        email = email_data[0]
        preprocessed_body = self.remove_html_tags(email['Body'])
        preprocessed_body = self.remove_links(preprocessed_body)

        processed_email_data = {
            'From': email['From'],
            'Date': email['Date'],
            'Subject': email['Subject'],
            'Labels': email.get('Labels', []),  # Include the 'Labels' key in the processed data
            'Body': preprocessed_body,
        }
        return processed_email_data




