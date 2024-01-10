import os
import configparser
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI

from mail import EmailClientAzure
import mail_init

# Get credentials from config.cfg
config = configparser.ConfigParser()
config.read('config.cfg')

# Set your Outlook email and password (or App Password if 2FA is enabled)
email_address = config['imap']['email']
email_password = config['imap']['pw']

# Default Azure connection key
default_azure_key = os.getenv("AZURE_KEY", "azure1")

azure_connection = mail_init.connect_to_azure(config[default_azure_key])
if isinstance(azure_connection, str):
    # If the connection is a string, it is an error message. Print it and exit.
    print(azure_connection)
    exit()

email_client = EmailClientAzure(azure_connection)

# Connect to MongoDB
db_hanlder = AsyncIOMotorClient(config['mongo']['connection_local'])
# db_hanlder = AsyncIOMotorClient(config['mongo']['connection_string'])
db_client = db_hanlder["broker"]

# Instantiate openai client
openai_client = AsyncOpenAI(
    api_key=config['openai']['api_key'],
)