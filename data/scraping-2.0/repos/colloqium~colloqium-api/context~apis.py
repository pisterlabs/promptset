from twilio.rest import Client
from sendgrid import SendGridAPIClient
import os
import openai
from dotenv import load_dotenv

load_dotenv()
# Your Twilio account credentials
account_sid = os.environ['twilio_account_sid']
auth_token = os.environ['twilio_auth_token']
segment_write_key = os.environ['WRITE_KEY']
twilio_messaging_service_sid = os.environ['TWILIO_MESSAGING_SERVICE_SID']

base_url = os.environ['BASE_URL']

# The webhook URL for handling the call events
call_webhook_url = base_url+"/twilio_call"

# The webhook URL for handling the message events
message_webhook_url = base_url+"/interaction_callback"

# Create a Twilio client object
twilio_client = Client(account_sid, auth_token)

# create SendGrid client object
sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))

# set OpenAi Key for GPT4
openai.api_key = os.environ['OPENAI_APIKEY']