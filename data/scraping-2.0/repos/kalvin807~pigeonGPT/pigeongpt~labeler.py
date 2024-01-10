import os
import re

import openai
import structlog
import tiktoken
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_random_exponential

from pigeongpt.provider.gmail import Email

log: structlog.stdlib.BoundLogger = structlog.get_logger()
openai.api_key = os.getenv("OPENAI_API_KEY")
enc = tiktoken.get_encoding("cl100k_base")

LABELS = ["Alert", "Bill", "Booking", "Newsletter", "Promotion"]
MAX_EMAIL_TOKEN_USAGE = 4000
CHAT_GPT_MODEL = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0, model_name=CHAT_GPT_MODEL)  # type: ignore
text_splitter = RecursiveCharacterTextSplitter().from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=50
)
summary_prompt = PromptTemplate(
    template="Please read the following text and provide a concise summary that captures the main ideas, while retaining important context, warnings, and other key details that would be relevant for classifying the email: \n\n{text}.",
    input_variables=["text"],
)
combine_summary_prompt = PromptTemplate(
    template="Write summary based on the data below, start with the type of article: \n\n{text}",
    input_variables=["text"],
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def get_gpt_response(prompt, temperature=0.0):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )


def estimate_token_size(prompt):
    return len(enc.encode(prompt))


def split_by_size(content_string, size=1000):
    return [content_string[i : i + size] for i in range(0, len(content_string), size)]


def summarise_content(content_string: str):
    splitted_texts = text_splitter.split_text(content_string)
    docs = [Document(page_content=t) for t in splitted_texts]
    chain = load_summarize_chain(
        llm,
        map_prompt=summary_prompt,
        combine_prompt=combine_summary_prompt,
        chain_type="map_reduce",
        verbose=True,
    )
    return chain.run(docs)


def make_label_prompt(email: Email) -> str:
    return f"""
    Given the following email:
    Subject: {email.subject}
    Sender: {email.sender}
    Content: {email.content}
    Determine the nature of this email.
    The possible labels are {",".join(f"'{label}'" for label in LABELS)} or 'unknown'.
    Only reply the label and wrap the label in '###'.
    """


def preprocess_email(email: Email):
    prompt = make_label_prompt(email)
    if estimate_token_size(prompt) > MAX_EMAIL_TOKEN_USAGE:
        email.content = summarise_content(email.content)
    return email


def label_email(email: Email):
    processed_email = preprocess_email(email)
    prompt = make_label_prompt(processed_email)
    response = get_gpt_response(prompt)
    log.debug("label email", prompt=prompt, response=response)
    response_message = response.choices[0].message.content.strip()
    match = re.search(r"###(.*)###", response_message)
    if match:
        label = match.group(1)
        if label not in LABELS:
            return "unknown"
        return label
    else:
        raise ValueError("No match found")
