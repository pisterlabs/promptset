from flask import Flask, Response, request
from threading import Thread
from slackeventsapi import SlackEventAdapter
from slack import WebClient

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
import nest_asyncio
nest_asyncio.apply()

##################### IMPORTANT NOTE: ########################
### These keys are not visible and should be hidden in a .env file

OPENAI_KEY = ""
SLACK_SIGNING_SECRET = ""
SLACK_BOT_TOKEN = ""
SLACK_BOT_ID = ""


##################################################
########## Chatbot and llm model setup ###########
##################################################


llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_KEY)

# Split up data into managable chunks and provide it to the model
# Retain memory of previous messages within the same server activation period
# Based on Langchain documentation and tutorials

df = pd.read_csv('ProductList.csv')
loader = DataFrameLoader(df, page_content_column = "Product Info")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0, separator = "Document(", )
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_KEY)
docsearch = Chroma.from_documents(texts, embeddings)

# keeps the last 4 interactions to avoid going over the maximum token length
memory = ConversationBufferWindowMemory(k = 4, return_messages = True)

qa = RetrievalQA.from_chain_type(llm, chain_type = "stuff", memory = memory, retriever = docsearch.as_retriever())


##################################################
########### Slack request and messages ###########
##################################################


# App creation and routing based on https://github.com/Saurav-Shrivastav/Slackbot-tutorial

app = Flask(__name__)
slack_client = WebClient(SLACK_BOT_TOKEN)

# Verify Slack requests and challenges
@app.route("/")
def event_hook(request):
    slack_json = json.loads(request.body.decode("utf-8"))
    if slack_json["token"] != SLACK_BOT_ID:
        return {"status": 403}

    if "type" in slack_json:
        if slack_json["type"] == "url_verification":
            res_json = {"challenge": slack_json["challenge"]}
            return res_json

    return {"status": 500}


slack_events = SlackEventAdapter(SLACK_SIGNING_SECRET, "/slack/events", app)


# Respond to every message that tags the chatbot
@slack_events.on("app_mention")
def handle_message(event):

    def reply(value):

        event = value
        message = event["event"]

        if message.get("subtype") is None:
            query = message.get("text")
            channel = message["channel"]

            # Run the user's query along with memory that stores previous messages
            convo_history = str(memory.load_memory_variables({}))
            answer = qa.run(convo_history + query)

            slack_client.chat_postMessage(channel = channel, text = answer)

    thread = Thread(target=reply, kwargs={"value": event})
    thread.start()

    return Response(status=200)


if __name__ == "__main__":
    app.run(port=3000)
