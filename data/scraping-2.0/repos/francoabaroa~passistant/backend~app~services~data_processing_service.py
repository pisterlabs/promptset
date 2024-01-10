from .data_retrieval_service import DataRetrievalService
from .data_storage_service import DataStorageService
import boto3
from enum import Enum
import json
import openai
import os
from os.path import basename
from ..prompts.assistant_manager import assistant_manager_prompt
from ..prompts.assistant_identifier import assistant_identifier_prompt
from pydub import AudioSegment
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
openai.api_key = os.getenv("OPENAI_API_KEY", "")
account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")

class Intent(Enum):
    SAVE = 'SAVE'
    VIEW_LISTS = 'VIEW_LISTS'
    VIEW_REMINDERS = 'VIEW_REMINDERS'


class DataProcessingService:
    def __init__(self):
        self.data_retrieval = DataRetrievalService()
        self.data_storage = DataStorageService()

    def identify_intent(self, original_text):
        try:
            intent = None
            conversation = [
                {
                    "role": "system",
                    "content": assistant_identifier_prompt
                },
                {
                    "role": "user",
                    "content": original_text
                }
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation,
            )

            assistant_reply = response['choices'][0]['message']['content']
            request_type = json.loads(assistant_reply).get('requestType', {})

            if request_type == 'viewReminders':
                intent = Intent.VIEW_REMINDERS
            elif request_type == 'viewLists':
                intent = Intent.VIEW_LISTS
            elif request_type == 'save':
                intent = Intent.SAVE

            return intent
        except Exception as e:
            print(f"Error in identify_intent: {str(e)}")
            return None

    def process_text(self, original_text, source, from_number):
        try:
            intent = self.identify_intent(original_text)

            if intent == Intent.VIEW_LISTS:
                list_names_and_items = self.data_retrieval.get_all_lists(from_number)
                message = "\n\n".join([f'- {list["name"]}\n' + '\n'.join([f'    - {item["name"]}: {item["quantity"]}' for item in list['items']]) for list in list_names_and_items])

                return json.dumps({
                    "responseMessage": message
                })
            elif intent == Intent.VIEW_REMINDERS:
                reminder_names = self.data_retrieval.get_all_reminder_names(from_number)
                message = "\n\n".join([f'- {reminder["name"]}' for reminder in reminder_names])

                return json.dumps({
                    "responseMessage": message
                })
            elif intent == Intent.SAVE:
                conversation = [
                    {
                        "role": "system",
                        "content": assistant_manager_prompt
                    },
                    {
                        "role": "user",
                        "content": original_text
                    }
                ]

                response = openai.ChatCompletion.create(
                    # model="gpt-3.5-turbo",
                    model="gpt-4",
                    messages=conversation,
                )

                # Extract the assistant's reply
                assistant_reply = response['choices'][0]['message']['content']
                json_assistant_reply = json.loads(assistant_reply)
                message_type = json_assistant_reply.get('messageType', {})

                if message_type == 'reminder':
                    self.data_storage.store_reminder(json_assistant_reply, original_text, source)
                elif message_type == 'list':
                    self.data_storage.store_list(json_assistant_reply, original_text, source)

                return str(assistant_reply)


        except Exception as e:
            print(f"Error in process_text: {str(e)}")
            return None

    def process_audio(self, media_url, from_number):
        parsed = urlparse(media_url)
        filename = basename(parsed.path)

        # Retrieve the file from the url
        response = requests.get(media_url, stream=True, auth=HTTPBasicAuth(account_sid, auth_token))
        if response.status_code == 200:
            with open(filename, 'wb') as audio_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        audio_file.write(chunk)

            # Convert the audio file to a supported format for OpenAI Whisper
            filename_without_ext = os.path.splitext(filename)[0]
            converted_filename = filename_without_ext + ".wav"

            audio = AudioSegment.from_file(filename)
            audio.export(converted_filename, format="wav")

            with open(converted_filename, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)

            # Process transcript
            source = 'VOICE NOTE'
            processed_text = self.process_text(transcript.text, source, from_number)

            # Remove the original and converted audio files
            os.remove(filename)
            os.remove(converted_filename)

            return processed_text

    def process_image(self, media_url, from_number):
        # Create a session using your AWS credentials
        boto3.setup_default_session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)

        # Create a Textract client
        textract = boto3.client(service_name='textract')

        # Download the image
        response = requests.get(media_url)
        img = response.content  # Get the image content

        # Use Textract to process the image and extract the text
        response = textract.detect_document_text(Document={'Bytes': img})
        extracted_text = ""

        # Extract detected text
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                extracted_text += item["Text"] + "\n"

        extracted_text = extracted_text.strip()

        # Process extracted text
        source = 'IMAGE'
        processed_text = self.process_text(extracted_text, source, from_number)
        return processed_text

    def process_document(self, media_url):
        # Use gpt3.5 to store safely in database
        # Your document processing code goes here
        print("Processing document from: {}".format(media_url))

    def process_unsupported(self, media_url):
        # Use gpt3.5 to store safely in database
        # Code to handle unsupported media types goes here
        print("Cannot process unsupported file type from: {}".format(media_url))
