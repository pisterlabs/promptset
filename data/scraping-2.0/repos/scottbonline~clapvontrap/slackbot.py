import os
from openai import OpenAI
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_bolt import App
import logging
#from langchain.chains import LLMChain
#from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI as lcOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import random
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb
#from chromadb.config import Settings
import re
import time
import spacy
import json
from agent_ability import ability_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
# openai auth token is pulled from system env OPENAI_API_KEY

# init openai using langchain
chat = ChatOpenAI(
    # openai_api_key=os.environ["OPENAI_API_KEY"],
    # openai_api_base = "http://192.168.1.59:1234/v1",
    temperature=0.7,
    # model='gpt-3.5-turbo'
    model="gpt-4-1106-preview"
    # model = "local-model"
)
llm = lcOpenAI()
openai_client = OpenAI()

# init Chroma
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
# chroma_collection = chroma_client.get_or_create_collection("20char")
chroma_collection = chroma_client.get_collection("10word")

# init Slack Event API & Web API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)

friendly_ai = """You are a friendly AI who is trained in analyzing json to be able to summarize
the contents in relation to what as user is asking. Pay careful attention to who said things, and what
sentiment they used. If the answer is not clear, make up something creative based on the context"""

default_ai_1 = """Imagine you're asked a hypothetical or personality based question about how a certain person 
would react in a certain scenario, like being on a deserted island, or are they more positive or negative 
about life. Keeping in mind that person's messages based from the provided context, craft a creative, 
humorous response that playfully exaggerates their traits. You must always give an answer. Do not complain 
about needing additional context"""

generic_ai = """You are an unhelpful AI that doesn't like to be disturbed with questions. If the question doesn't have an answer, express your displeasure."""

default_ai = """You are a creative storyteller who performs the following tasks:

Task #1:
Summarize in less than 100 words everything in the "Context" section
Task #2:
Imagine you're asked a hypothetical or personality based question about how a certain person 
would react in a certain scenario, like being on a deserted island, or are they more positive or negative 
about life. Keeping in mind that person's messages based from the provided context, craft a creative, 
 response that exaggerates their traits. You must always give an answer. Do not complain 
about needing additional context. Do not mention a desert island in your response.

Your response should be formatted as follows:

Summary: <summary of context>
Analysis: <creative story with a dark twist based on the question>
"""

messages = [
    SystemMessage(content=default_ai),
]

messages_generic = [
    SystemMessage(content=generic_ai),
]

def valid_users():
    file_path = "usermap.json"
    with open(file_path, "r") as file:
        data = json.load(file)
        values_list = list(data.values())
        values_list = [name.lower() for name in values_list]

    return values_list


def query_chroma(query, subject=None):
    logging.info(f"Query: {query}, Sender: {subject}")
    if subject:
        # FIX can clean this upper case nonsense up on next import of RAG
        if not subject[0].isupper():
            subject = subject[0].upper() + subject[1:]
        c_results = chroma_collection.query(
            query_texts=[query],
            n_results=10,
            # use this to search metadata keys
            where={"sender": subject},
            # where_document={"$contains":"search_string"}
        )
    else:
        c_results = chroma_collection.query(
            query_texts=[query],
            n_results=10,
            # use this to search metadata keys
            # where={"sender": sender},
            # where_document={"$contains":"search_string"}
        )
    # clean results
    raw_results = c_results.get("metadatas") + c_results.get("documents")
    results = {}
    for i in range(len(raw_results[1])):
        results[i] = {"metadata": raw_results[0][i], "message": raw_results[1][i]}

    return results


def augment_prompt(query: str, sender=None):
    # get top X results from Chroma
    if sender:
        logging.info(f"Subject Detected")
        source_knowledge = query_chroma(query, sender)
        logging.info(f"Source Knowledge:: {source_knowledge}")
    else:
        logging.info(f"Subject NOT Detected")
        source_knowledge = query_chroma(query)
        logging.info(f"Source Knowledge:: {source_knowledge}")

    # feed into an augmented prompt
    augmented_prompt = f"""{default_ai}

    Context:
    {source_knowledge}

    """
    return augmented_prompt


def image_create(context_from_user):
    logging.info(f"Generate image using:: {context_from_user}")
    aiimage = openai_client.images.generate(
        prompt=context_from_user,
        model="dall-e-3",
        n=1,
        size="1024x1024",
    )

    return aiimage


def get_subject(query):
    if not query[0].isupper():
        logging.info(f"add uppercase: {query}")
        context_from_user = query[0].upper() + query[1:]

    logging.info("Start Subject Detection")
    # Load the English model
    nlp = spacy.load("en_core_web_sm")

    # Process the sentence
    doc = nlp(query)
    # generate valid users
    valid_names = valid_users()
    # Find the subject
    for token in doc:
        # 'nsubj' stands for nominal subject; 'nsubjpass' stands for passive nominal subject
        logging.info(f"Subject Details:: {token.text, token.dep_}")
        if token.dep_ in ("nsubj", "nsubjpass", "npadvmod", "dobj"):
            if token.text.lower() in valid_names:
                logging.info(f"Subject Detected:: {token.text, token.dep_}")
                return token.text
    logging.info(f"Subject NOT Detected")
    return None


def chat_response(context_from_user):
    # formatting to help with NLP
    if not context_from_user[0].isupper():
        context_from_user = context_from_user[0].upper() + context_from_user[1:]
    logging.info(f"add uppercase: {context_from_user}")

    subject = get_subject(context_from_user)

    if not subject:
        prompt = [
            SystemMessage(content=generic_ai),
            HumanMessage(content=f"Question: {context_from_user}"),
        ]
    else:
        prompt = [
            SystemMessage(content=augment_prompt(context_from_user, subject)),
            HumanMessage(content=f"Question: {context_from_user}"),
        ]

    logging.info(f"Sending finalized prompt:: {prompt}")
    ai_response = chat(prompt)
    logging.info(f"Recived response: {ai_response}")

    return ai_response


# This gets activated when the bot is tagged in a channel
@app.event("app_mention")
def handle_message_events(body):

    context_from_user = str(body["event"]["text"]).split("> ")[1]

    # Let thre user know that we are busy with the request
    response = client.chat_postMessage(
        channel=body["event"]["channel"],
        # thread_ts=body["event"]["event_ts"],
        text=f"beep, boop: " + context_from_user,
    )

    logging.info(f"Check Query for Image Request:: {context_from_user}")

    if context_from_user.startswith("i:"):
        logging.info("Image Search Detected")
        ai_response = image_create(context_from_user)

        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            # thread_ts=body["event"]["event_ts"],
            text=ai_response.data[0].model_dump()["url"],
        )
    elif context_from_user.startswith("p:"):
        logging.info("BETA Personality Detected")
        ai_response = ability_check(context_from_user)

        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            # thread_ts=body["event"]["event_ts"],
            text=ai_response,
        )
    else:
        logging.info("No Image Search Detetected")
        ai_response = chat_response(context_from_user)

        response = client.chat_postMessage(
            channel=body["event"]["channel"],
            # thread_ts=body["event"]["event_ts"],
            text=ai_response.content,
        )

# this listens to all messages in all channels
@app.event("message")
def handle_message_events(body, logger):
    if "text" in body["event"]:
        context_from_user = str(body["event"]["text"])
        chance = random.randint(1, 30)
        length = len(context_from_user)
        logging.info(
            f"Random response check:: Context: {context_from_user}, Chance:{chance}, Length:{length}"
        )

        if (
            chance > 25
            and length > 8
            and context_from_user[-1] == "?"
            and "<@U04PUPJ04R0>" not in context_from_user
        ):
            logging.info("Random response activated")
            ai_response = chat_response(context_from_user)

            response = client.chat_postMessage(
                channel=body["event"]["channel"],
                # thread_ts=body["event"]["event_ts"],
                text=ai_response.content,
            )
    else:
        logger.info(f"No 'text' key found:: {body}")

if __name__ == "__main__":
    try:
        # start slack handler
        SocketModeHandler(app, SLACK_APP_TOKEN).start()
    except Exception as e:
        print(e)
