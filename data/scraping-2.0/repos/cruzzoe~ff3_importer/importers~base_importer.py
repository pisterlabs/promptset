
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import unicodedata
from abc import ABC, abstractmethod

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

secret_id=os.getenv("GC_SECRET_ID")
secret_key=os.getenv("GC_SECRET_KEY")
# log_file = '' os.getenv("LOG_LOCATION")

# handler = RotatingFileHandler(
#     log_file,
#     maxBytes=1024 * 1024,
#     backupCount=5,
# )


def configure_logger():
    formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    )
    # handler.setFormatter(formatter)

    # Create a StreamHandler to log messages to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
        )
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(stream_handler)
    # logger.addHandler(handler)

    # should_roll_over = os.path.isfile(log_file)
    # if should_roll_over:  # log already exists, roll over!
    #     handler.doRollover()

    logger.setLevel(logging.INFO)
    return logger

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOKEN = os.getenv("TOKEN")
HOME_IP = os.getenv("HOME_IP")
GOTIFY_TOKEN=os.getenv('GOTIFY_TOKEN')
class BaseImporter(ABC):

    def __init__(self, imports_dir):
        self.import_dir = imports_dir
        self.logger = configure_logger() 
        self.logger.info('Init class' + self.__class__.__name__)

    def create_hash(self, row):
        result = hashlib.sha256(row.encode())
        return result.hexdigest()

    def get_token(self):
        try:
            self.logger.info('Retrieving GC token')
            curl_command = f"""
            curl -X POST "https://bankaccountdata.gocardless.com/api/v2/token/new/" \
            -H "accept: application/json" \
            -H  "Content-Type: application/json" \
            -d '{{"secret_id":"{secret_id}", "secret_key":"{secret_key}"}}'
            """
            process = subprocess.run(curl_command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
            token = json.loads(process.stdout)
            return token['access']
        except:
            self.notify('FF3_IMPORT', 'Error retrieving token from GoCardless API')
            raise

    def get_data(self, account):
        try:
            self.logger.info('Getting Data from GoCardless API')
            GC_TOKEN = self.get_token()
            curl_command = f"""
            curl -X GET "https://bankaccountdata.gocardless.com/api/v2/accounts/{account}/transactions/" \
            -H  "accept: application/json" \
            -H  "Authorization: Bearer {GC_TOKEN}"
            """
            process = subprocess.run(curl_command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
            data = json.loads(process.stdout)
            self.logger.info(data)
            booked = data['transactions']['booked']
            self.logger.info('Data Downloaded')
            return booked
        except:
            self.notify('FF3_IMPORT', 'Error downloading data from GoCardless API')
            raise

    def copy_template(self):
        self.logger.info('Copying template')
        class_name = self.__class__.__name__
        output_path = os.path.join(self.import_dir, class_name + '.json')
        script_path = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(script_path)
        config_path = os.path.join(parent_dir, 'import_configs', class_name + "_config.json")
        shutil.copyfile(config_path, os.path.join(self.import_dir, output_path))
        self.logger.info(f'JSON Import config copied to import directory: {self.import_dir}')

    def empty_imports(self):
        """Empty the import directory"""
        if os.path.exists(self.import_dir):
            shutil.rmtree(self.import_dir)
        os.makedirs(self.import_dir, exist_ok=True)
        
    
    def upload_to_firefly(self):
        # TODO check if any data was actually imported. If not, then don't notify!
        try:
            self.logger.info('uploading to firefly')
            completed_process = subprocess.run([
            "docker", "run",
            "--rm",
            "-v", f"{self.import_dir}:/import",
            "-e", f"FIREFLY_III_ACCESS_TOKEN={TOKEN}",
            "-e", "IMPORT_DIR_ALLOWLIST=/import",
            "-e", f"FIREFLY_III_URL={HOME_IP}:8995",
            "-e", "WEB_SERVER=false",
            "fireflyiii/data-importer:latest"
            ], capture_output=True, text=True)
            self.logger.info("Output: %s", completed_process.stdout)
            self.logger.info("Error: %s}" ,completed_process.stderr)
            self.notify('FF3_IMPORT', 'Data imported sucessfully for ' + self.__class__.__name__)
        except:
            self.logger.error('Error uploading to firefly')
            self.notify('FF3_IMPORT', 'Error uploading to firefly for ' + self.__class__.__name__)
            raise

    
    def notify(self, header, message):
        """Send notification to gotify"""
        cmd = f'curl "http://{HOME_IP}:8991/message?token={GOTIFY_TOKEN}" -F "title=[{header}]" -F "message"="{message}" -F "priority=5"'
        subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    def to_csv(self, df):
        try:
            class_name = self.__class__.__name__
            output_path = os.path.join(self.import_dir, class_name + '.csv')
            rows = len(df)
            self.logger.info(f"Number of rows in df to save to csv: {rows}")
            df.to_csv(output_path, encoding="utf-8")
            self.logger.info(f'Saved to path {output_path}')
        except:
            self.logger.error('Error saving to csv')
            self.notify('FF3_IMPORT', 'Error saving to csv for ' + self.__class__.__name__)
            raise

    def is_japanese(self, string):
        if bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u30FC]', string)):
            return string
        else:
            return ''
    
    def translate(self, text):
        # use chatgpt to translate text from japanese to english
        try:
            prompt =  f"Translate to english '{text}' from the perspective of converting a shopping merchant name."
            response = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=prompt, temperature=0.2)
            translated_text = response.choices[0].text.strip()
            self.logger.info(f'Translate: {text} to {translated_text}')
            return translated_text.replace('"', '')
        except:
            self.logger.error('Error translating text')
            self.notify('FF3_IMPORT', 'Error translating text for ' + self.__class__.__name__)
            raise

    def normalize_text(self, text):
        return unicodedata.normalize('NFKC', text)

    def handle_pure_japanese(self, df):
        # Where Notes column contains a value, use ChatGpt API to translate the value to English and replace the value in the Name column with the translated value.
        df['Notes'] = df['Name'].apply(self.is_japanese)
        # for rows that have a value in Notes column, send this value to translate function and replace the value in the Name column with the translated value.
        df.loc[df['Notes'] != '', 'Name'] = df.loc[df['Notes'] != '', 'Notes'].apply(self.translate)    
        return df
    
    def apply_normalization(self, df):
        df['Name'] = df['Name'].apply(self.normalize_text)
        return df
    
    def create_unique_id(self, df):
        """Create a has based on Name, amount and data strings."""
        df['unique_base'] = df['Notes'].fillna(df['Name'])
        df['unique_id'] = df.apply(lambda row: self.create_hash(row['unique_base'] + row['Amount'] + row['Date']), axis=1)
        return df
    
        
    @abstractmethod
    def run(self):
        pass


