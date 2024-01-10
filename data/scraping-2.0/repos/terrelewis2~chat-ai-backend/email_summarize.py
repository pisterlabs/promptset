from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText

import os
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()


def summarize_thread(thread_id):
    # Load the credentials
    creds = Credentials.from_authorized_user_file('token.json')

    # Build the service
    service = build('gmail', 'v1', credentials=creds)

    # fetch the thread
    thread_details = get_thread_details(service, thread_id)
    headers = thread_details['payload']['headers']
    subject = [i['value'] for i in headers if i["name"] == "Subject"][0]
    msg_id_header = [i['value'] for i in headers if i["name"] == "Message-ID"][0]
    request_email_id = [i['value'] for i in headers if i["name"] == "From"][0]

    # fetch the body
    thread = thread_details['payload']['parts'][0]['body']['data']

    # Decode the base64url encoded string
    thread = thread.replace("-", "+").replace("_", "/")
    decoded_data = base64.b64decode(thread)

    # Print the decoded message
    print('Decoded data: ', decoded_data)

    # Convert bytes to string
    decoded_data_str = decoded_data.decode('utf-8')

    # Replace non-breaking space characters and new line characters
    clean_data = decoded_data_str.replace('\xe2\x80\xaf', ' ').replace('\r\n', '\n')

    print(clean_data)

    summary = get_thread_summary(clean_data)

    # The email and message you want to send
    sender = "threadsummarizer@gmail.com"
    to = request_email_id
    message_text = summary

    # Create the message
    message = create_message(sender, subject, msg_id_header, to, message_text, thread_id)

    # Send the message
    data = send_message(service, 'me', message)

    print(data)

    return data


def get_thread_details(service, thread_id):
    # Get the email
    message = service.users().messages().get(userId='me', id=thread_id).execute()
    print(message)

    return message


def get_thread_summary(thread):
    key: str = os.environ.get("OPENAI_API_KEY")
    print(key)
    chat_gpt = ChatOpenAI(openai_api_key=key, temperature=0, model_name="gpt-4")
    template = """
    Write a concise bullet list summary of the email thread between the participants":

    {text}

    Concise summary using markdown:"""

    prompt = PromptTemplate(template=template, input_variables=["text"])
    summary_chain = load_summarize_chain(
        chat_gpt, chain_type="stuff", verbose=True, prompt=prompt
    )

    docs = [Document(page_content=thread)]
    summary_result = summary_chain.run(docs)
    print(summary_result)
    return summary_result
    # key: str = os.environ.get("OPENAI_API_KEY")
    # print(key)
    # chat_gpt = ChatOpenAI(openai_api_key=key, temperature=0, model_name="gpt-4")
    #
    # template = """
    # You have to summarise the email thread body below: "
    #
    # {thread_body}
    #
    # Include the following information:
    # 1. The participants in the email thread along with their email ids.
    # 2. The subject of the email thread.
    # 3. The number of emails exchanged in the email thread.
    # 4. The number of emails sent by each participant in the email thread.
    # 5. The topics discussed in the email thread.
    # 6. The sentiment of each email in the email thread.
    #
    # Summary:\n
    # """
    # prompt = PromptTemplate(input_variables=["thread_body"], template=template)
    #
    # location_chain = LLMChain(llm=chat_gpt, prompt=prompt)
    #
    # response = location_chain(thread)
    #
    # print('Summary:', response['text'])
    #
    # return response['text']


def create_message(sender, subject, msg_id_header, to, message_text, thread_id):
    """Create a message for an email.

    Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.
    thread_id: The thread id of the email message to reply to.

    Returns:
    An object containing a base64url encoded email object.
    """
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    message['In-Reply-To'] = msg_id_header
    message['References'] = msg_id_header
    raw_message = base64.urlsafe_b64encode(message.as_bytes())
    raw_message = raw_message.decode()
    body = {'raw': raw_message, 'threadId': thread_id}
    return body


def send_message(service, user_id, message):
    """Send an email message.

    Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me" can be used to indicate the authenticated user.
    message: Message to be sent.

    Returns:
    Sent Message.
    """
    try:
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        print('Message Id: %s' % message['id'])
        return message
    except HttpError as error:
        print('An error occurred: %s' % error)


# summarize_thread('188a3c309526bae4')
