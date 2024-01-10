from utilities.chromadb_manager import *
from utilities.agent import *

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain.memory import ConversationBufferWindowMemory

from langchain.agents import (
    AgentExecutor,
    AgentType,
    initialize_agent,
    Tool
)

from langchain.agents.agent_toolkits import (
    create_retriever_tool,
    create_conversational_retrieval_agent
)

from langchain.chains import ConversationalRetrievalChain

system_message="Your role is to be a helpful assistant with a friendly, "\
    "understanding, patient, and user-affirming tone. You should: "\
    "explain topics in short, simple sentences; "\
    "keep explanations to 2 or 3 sentences at most. "\
    "If the user provides affirmative or brief responses, "\
    "take the initiative to continue with relevant information. "\
    "Check for user understanding after each brief explanation "\
    "using varied and friendly-toned questions. "\
    "Use ordered or unordered lists "\
    "(if longer than 2 items, introduce them one by one and "\
    "check for understanding before proceeding), or simple text in replies. "\
    "Provide examples or metaphors if the user doesn't understand. "\
    "If you need to use trusty tools to give answers, "\
    "please always use the indications provided before to interact with the user "

print("START")

load_dotenv()
model_name=os.environ.get('FINE_TUNED_MODEL')
print(model_name)
temperature=0
k=3
language_code="en"
overwrite=True
local_file_name="./files/jokerbirot_space_musician_en.txt"
save_directory='./files'
persist_directory = 'chroma'

conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", #needs to be present
    ai_prefix="AI Assistant",
    k=k, #how many interactions can be remembered
    return_messages=True
)
llm=ChatOpenAI(
    model_name=model_name,
    temperature=temperature,
    streaming=True
)

embedding = OpenAIEmbeddings()

local_file=save_file(
    location=local_file_name,
    language_code=language_code
)

vectordb=create_vectordb_from_file(
    filename=local_file,
    persist_directory=persist_directory,
    embedding=embedding,
    overwrite=overwrite
)

conversation = ConversationalRetrievalChain(
    llm=OpenAI(temperature=0),
    memory=conversational_memory,
    verbose=True,
    retriever=retriever
)

prompt="Tell me something about jokerbirot"
response=conversation.predict(prompt)
print("RESPONSE: "+response)

# prompt="Tell me something about what is quantum bit"
# response = agent.run(prompt)
# print("RESPONSE: "+response)