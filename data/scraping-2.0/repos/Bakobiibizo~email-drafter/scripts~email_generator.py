import os
import subprocess
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def run_email_draft():
    with open("./docs/input.txt", "r", encoding="utf-8") as f:
        doc = f.read()

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(doc)
    chat = ChatOpenAI(temperature=0)
    with open("./docs/input.json", "r", encoding="utf-8") as f:
        examples = json.load(f)
    messages = [
        SystemMessage(
            content="""
        You are a sales guru. You are a specialist in SaaS marketing. You have been hired by a start up called Move Right. You are drafting sales emails to go out to Move Rights best prospects. 
        Here are some examples of support ticket requests:{examples}.
        Always sign off as Richard
        Use a more personal tone as I have a personal relationship with the customer we have been working closely together for the last 6 months.
        Do not apologize if there was no fault on our side. Instead, thank the customer for their patience and understanding.
        Do not use overtly corporate language. Use a more personal tone.
        Provide your responses in valid JSON format with this schema:
    
        "response":[{
        "draft_email":"[DRAFT_EMAIL_RESPONSE]",
        "user_email":"[USER_EMAIL_RESPONSE]",
        "draft_email_subject":"[DRAFT_EMAIL_SUBJECT_RESPONSE]"
        }]
    
        review your response and ensure it is valid JSON before submitting.
        """
        ),
        HumanMessage(content=doc),
    ]
    content = chat(messages).content
    # print(content)
    json_response = json.loads(content)
    print(json_response)
    user_email = json_response["response"][0]["user_email"]
    draft_email = json_response["response"][0]["draft_email"]
    draft_email_subject = json_response["response"][0]["draft_email_subject"]
    print(user_email, draft_email_subject, draft_email)
    with open("./docs/output.txt", "w", encoding="utf-8") as f:
        f.write(user_email + "\n" + draft_email_subject + "\n" + draft_email)

    def send_draft_email(user_email, draft_email, draft_email_subject):
        subprocess.Popen(
            [
                "python",
                "./scripts/gmail.py",
                "--draft_email",
                draft_email,
                "--user_email",
                user_email,
                "--draft_email_subject",
                draft_email_subject,
            ],
            bufsize=1024,
        )

    send_draft_email(
        user_email=user_email,
        draft_email=draft_email,
        draft_email_subject=draft_email_subject,
    )
