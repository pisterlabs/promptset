from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json, os

load_dotenv()

cloud_config = {"secure_connect_bundle": "secure-connect-choose-your-adventure.zip"}


with open("token.json") as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]
ASTRA_DB_KEYSPACE = "database"
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")


auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
    print(row[0])
else:
    print("An error occurred.")


message_history = CassandraChatMessageHistory(
    session_id="history1",
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    ttl_seconds=3600
)

# new set of memory for the new instance of the game
message_history.clear()


template = """
You're now the navigator of an extraordinary expedition through the Temporal Traverse. 
A historian named Alex embarks on a quest to safeguard the Timeline Key. 
Your task is to navigate Alex through various time periods, weaving a narrative filled with challenges, choices, and their consequences. 
Your choices will craft diverging paths, ultimately shaping the fate of the timeline. Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining Alex's fate. 

Here are the rules for our journey:
1. Start by asking the player to select a time-travel device that will determine their mode of transportation.
2. Present several paths that lead to success.
3. After three decision points. Include paths that lead to death, Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game
4. You must provide atleast three choices for each decision point.

Here's the chat history. Use this conversation as context for the unfolding story: {chat_history}
Human: {human_input}
AI:
"""



cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=cass_buff_memory
)

choice = "start"

while True:
    response = llm_chain.predict(human_input=choice)
    print(response.strip())

    if "The End." in response:
        break

    choice = input("Your reply: ")

# test run
# response = llm_chain.predict(human_input="start the game")
# print(response)