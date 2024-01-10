from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import os
import base64

import json

# own imports
import MLStripper
import OpenAIConnector

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

class MailConnector:
    SCOPES= ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']
    
    label_gpt: str = os.getenv('LABEL_GPT_SUMMARIZED')
    importance_label: str = os.getenv('IMPORTANCE_LABEL')
    unimportant_label: str = os.getenv('UNIMPORTANT_LABEL')
 
    openaiconnector = OpenAIConnector.OpenAiConnector()
    
    creds = None
    
    gpt_prompt = '''
                    You are a helpful assistant.
                    I will give you my latest mail and you will rate its importance that I should answer it directly from 1 to 5.
                    Your answer will contain the string 'Importance: x' where x is the importance. If the mail contains advertisment or simple information, you will rate it with 1.
                    You will also summarize the mail in a few sentences. The summary will contain the string 'Summary: x' where x is the summary. The summary should contain the sender of the mail.
                    You will also give me a suggestion for a response. The response will contain the string 'Response: x' where x is the response.
                    If the mail was in English, your answer and summary will be in English. If the mail was mostly German, your answer and summary will be in German. Importance, summary and response will always be in English.

                    Mail:
                    '''
    
    def __init__(self):
                # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        creds = None
        if os.path.exists('token.json'):
            logging.info("Token exists")
            creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logging.info("Refreshing token")
                creds.refresh(Request())
            else:
                logging.info("Creating new token")
                flow = InstalledAppFlow.from_client_secrets_file(
                    'src/credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        self.service = build('gmail', 'v1', credentials=creds)

    def __retreive_mail(self, maxResults= 5, positiveLabels = [], negativeLabels =[]):
        try:
            logging.info("get_mail started")
            # Call the Gmail API
        
            positiveLabelIds = []
            result = None
            
            # get label id of label in positiveLabels and set it to its id
            for positivelabel in positiveLabels:
                positiveLabelIds.append(self.__get_label_id(positivelabel))

            # negative querystring as query because gmail api does not support negative labels
            negativeQuerystring = ""
            for negativelabel in negativeLabels:
                negativeQuerystring += f"-label:{negativelabel} "
            result = self.service.users().messages().list(userId='me', labelIds=positiveLabelIds, maxResults=maxResults, q=negativeQuerystring).execute()
            logging.info("get_mail finished")            
            return result

        except HttpError as error:
            # TODO(developer) - Handle errors from Gmail API.
            logging.error(f'An error occurred: {error}')
            return None

    def get_mailContent(self, msg):
        
        msg = self.service.users().messages().get(userId='me', id=msg['id']).execute()
        #if msg['id'] not in self.summarized_mails and self.label_gpt not in msg['labelIds']:
            # extract email details
        headers = msg['payload']['headers']

        sender = self.__find_header_value(headers, 'From')
        send_date = self.__find_header_value(headers, 'Date')
        subject = self.__find_header_value(headers, 'Subject')
        body = self.__retreive_mail_body(msg)

        #use MLStripper to remove html tags
        body = MLStripper.clean_html(body)

        return sender, send_date, subject, body

    def get_mail(self, maxResults= 5, positiveLabels = [], negativeLabels =[]):        
       

        # create list of all mails

        result = self.__retreive_mail(maxResults, positiveLabels, negativeLabels)

        mail_list = []
        for msg in result.get('messages', []):
            sender, send_date, subject, body = self.get_mailContent(msg)
            
            # create email dictionary
            email_dict = {
                "from": sender,
                "send_date": send_date,
                "subject": subject,
                "body": body
            }
            logging.info("email_dict created. Sending to OpenAI")
            importance = 0
            summary = ""
            response = ""
            retry_count = 0
            while retry_count < 3:
                try:
                    response_text = self.openaiconnector.send_to_openai(self.gpt_prompt, email_dict)
                    importance, summary, response = self.openaiconnector.extract_info(response_text)
                    if importance > 0:
                        break
                except Exception as e:
                    logging.error(f"Error occurred while sending to OpenAI: {e}")
                retry_count += 1
            if importance == 0:
                logging.warning("Could not extract importance after 3 attempts.")
                continue

            response_dict = {
                "send_date": send_date,
                "sender": sender,
                "subject": subject,
                "importance": importance,
                "summary": summary,
                "response": response
            }
            #importance : int = int(''.join(filter(str.isdigit, str(importance))))
        
            if importance > 3:
                self.__add_label_to_mail(msg['id'], self.label_gpt, True)
            else:
                self.__add_label_to_mail(msg['id'], self.label_gpt, False)
                        
            mail_list.append(response_dict)

        if len(mail_list) < 1:
            return None
        return mail_list
    
    def __find_header_value(self, headers, name):
        for header in headers:
            if header['name'].lower() == name.lower():
                return header['value']
        return None

    def __retreive_mail_body(self, msg):
        if 'parts' in msg['payload']:
            for p in msg['payload']['parts']:
                if p['mimeType'] in ['text/plain', 'text/html']:
                    data = base64.urlsafe_b64decode(p['body']['data']).decode('utf-8')
                    return data
        else:
            if msg['payload']['mimeType'] in ['text/plain', 'text/html']:
                data = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
                return data
        
    
    def __add_label_to_mail(self, mail_id, label = None, gpt_importance=False):
        try:
            message = self.service.users().messages().get(userId='me', id=mail_id).execute()
            labels = message['labelIds']

            gpt_summarized_label_id = self.__get_label_id(self.label_gpt)
            modify_request_gpt = {'addLabelIds': [gpt_summarized_label_id], 'removeLabelIds': []}
            message = self.service.users().messages().modify(userId='me', id=mail_id, body=modify_request_gpt).execute()

            if gpt_importance:
                gpt_importance_label_id = self.__get_label_id(self.importance_label)
                modify_request_importance = {'addLabelIds': [gpt_importance_label_id], 'removeLabelIds': []}
                message = self.service.users().messages().modify(userId='me', id=mail_id, body=modify_request_importance).execute()
            else: 
                gpt_unimportant_label_id = self.__get_label_id(self.unimportant_label)
                modify_request_unimportant = {'addLabelIds': [gpt_unimportant_label_id], 'removeLabelIds': []}
                message = self.service.users().messages().modify(userId='me', id=mail_id, body=modify_request_unimportant).execute()

        except HttpError as error:
            logging.error(f'An error occurred: {error}')
            return


    def __get_labels(self):
                   
        labelsList = self.service.users().labels().list(userId='me').execute()

        return labelsList
        
    def __get_label_id(self, labelname):
        
        labels = self.__get_labels()
        #find label id of label with name label
        label_id = [label['id'] for label in labels['labels'] if label['name'] == labelname]
        if label_id:
            label_id = label_id[0]  # Take the first matching label ID
        else:
            label_id = None
        #label_id = labels['labels'][label]['id']
        return label_id
