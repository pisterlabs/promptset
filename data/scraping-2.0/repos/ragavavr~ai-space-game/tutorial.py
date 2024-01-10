from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate
import json

class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"

def print_user_input(text):
    print(f"{Colors.BLUE}Human:{Colors.RESET} {text}")

def print_ai_response(text):
    print(f"{Colors.GREEN}AI:{Colors.RESET} {text}")

def display_game_result(result):
    print(f"{Colors.GREEN}Game Result:{Colors.RESET}")
    print(result)
    print("The adventure concludes. Thanks for playing!")

# Add Zip File and JSON File LOCATION

cloud_config = {
    'secure_connect_bundle': 'secure-connect-file.zip'
}

with open("connect-token.json") as f:
    secrets = json.load(f)

# Add DB KeySpace and OpenAI API Key

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]
ASTRA_DB_KEYSPACE = ""
OPENAI_API_KEY = ""

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

message_history = CassandraChatMessageHistory(
    session_id="anything",
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    ttl_seconds=3600
)

message_history.clear()

cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

template = """
You are now the guide of an intergalactic journey in the Cosmos Explorer. 
A space traveler named Nova seeks the legendary Star Crystal. 
You must navigate her through challenges, choices, and consequences, 
dynamically adapting the tale based on the traveler's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining Nova's fate. 

Here are some rules to follow:
1. Start by asking the player to choose a type of spacecraft that will be used later in the game.
2. Have a few paths that lead to success in finding the Star Crystal.
3. Have some paths that lead to failure or danger in space. If the user faces a critical situation, generate a response that explains the outcome and ends in the text: "The End." I will search for this text to end the game.

Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

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

try:
    while True:
        response = llm_chain.predict(human_input=choice)
        print_ai_response(response.strip())

        if "The End." in response:
            display_game_result(response)
            break

        choice = input(f"{Colors.BLUE}Your reply:{Colors.RESET} ")

except KeyboardInterrupt:
    print("\nGame interrupted by user. Thanks for playing!")
