from PIL import Image
import streamlit as st
import openai
#exercise 11
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
#exercise 12
from langchain.memory import ConversationBufferWindowMemory
#exercise 13
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
import lancedb
import os
import tempfile
#exercise 15
import sqlite3
import pandas as pd
from datetime import datetime
#exercise 16
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
#Exercise 17
from langchain.agents import tool
import json

# os.environ["OPENAI_API_KEY"] = st.secrets["openapi_key"]
# openai.api_key = st.secrets["openapi_key"]

#Global ex 13
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)
#ex15
DB_NAME = os.path.join(WORKING_DIRECTORY, "default_db")

def ex15_initialise():
	# Create or check for the 'database' directory in the current working directory
	# Set DB_NAME to be within the 'database' directory at the top of main.py
	# Connect to the SQLite database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Conversation data table
	cursor.execute(
		"""
		CREATE TABLE IF NOT EXISTS data_table (
			id INTEGER PRIMARY KEY,
			date TEXT NOT NULL UNIQUE,
			username TEXT NOT NULL,
			chatbot_ans TEXT NOT NULL,
			user_prompt TEXT NOT NULL,
			tokens TEXT
		)
	"""
	)
	conn.commit()
	conn.close()

def ex15_collect(username, chatbot_response, prompt):
	# collect data from bot
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()
	now = datetime.now()  # Using ISO format for date
	tokens = len(chatbot_response) * 1.3
	cursor.execute(
		"""
		INSERT INTO data_table (date, username,chatbot_ans, user_prompt, tokens)
		VALUES (?, ?, ?, ?, ?)
	""",
		(now, username, chatbot_response, prompt, tokens),
	)
	conn.commit()
	conn.close()

# implementing data collection and displaying
def ex15():
	# initialise database first
	ex15_initialise()
	# collect some data
	ex15_collect("yoda", "I am Yoda. The Force is strong with you", "Who are you?")
	# display data
	# Connect to the specified database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Fetch all data from data_table
	cursor.execute("SELECT * FROM data_table")
	rows = cursor.fetchall()
	column_names = [description[0] for description in cursor.description]
	df = pd.DataFrame(rows, columns=column_names)
	st.dataframe(df)
	conn.close()

# smart agents accessing the internet for free
# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/search_and_chat.py
def ex16_agent_bot():
	st.title("🦜 LangChain: Chat with internet search")

	msgs = StreamlitChatMessageHistory()
	memory = ConversationBufferMemory(
		chat_memory=msgs,
		return_messages=True,
		memory_key="chat_history",
		output_key="output",
	)
	if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
		msgs.clear()
		msgs.add_ai_message("How can I help you?")
		st.session_state.steps = {}

	avatars = {"human": "user", "ai": "assistant"}
	for idx, msg in enumerate(msgs.messages):
		with st.chat_message(avatars[msg.type]):
			# Render intermediate steps if any were saved
			for step in st.session_state.steps.get(str(idx), []):
				if step[0].tool == "_Exception":
					continue
				with st.status(
					f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
				):
					st.write(step[0].log)
					st.write(step[1])
			st.write(msg.content)

	if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
		st.chat_message("user").write(prompt)

		llm = ChatOpenAI(
			model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, streaming=True
		)
		tools = [DuckDuckGoSearchRun(name="Search")]
		chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
		executor = AgentExecutor.from_agent_and_tools(
			agent=chat_agent,
			tools=tools,
			memory=memory,
			return_intermediate_steps=True,
			handle_parsing_errors=True,
		)
		with st.chat_message("assistant"):
			st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
			response = executor(prompt, callbacks=[st_cb])
			st.write(response["output"])
			st.session_state.steps[str(len(msgs.messages) - 1)] = response[
				"intermediate_steps"
			]

def upload_file_streamlit():
	def get_file_extension(file_name):
		return os.path.splitext(file_name)[1]

	st.subheader("Upload your docs")

	# Streamlit file uploader to accept file input
	uploaded_file = st.file_uploader("Choose a file", type=["docx", "txt", "pdf"])

	if uploaded_file:
		# Reading file content
		file_content = uploaded_file.read()

		# Determine the suffix based on uploaded file's name
		file_suffix = get_file_extension(uploaded_file.name)

		# Saving the uploaded file temporarily to process it
		with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
			temp_file.write(file_content)
			temp_file.flush()  # Ensure the data is written to the file
			temp_file_path = temp_file.name
		return temp_file_path
	
#exercise 13 - split and chunk, embeddings and storing in vectorstores for reference
def vectorstore_creator():
	# WORKING_DIRECTORY set above in the main.py
	# Process the temporary file using UnstructuredFileLoader (or any other method you need)
	embeddings = OpenAIEmbeddings()
	db = lancedb.connect(WORKING_DIRECTORY)
	table = db.create_table(
		"my_table",
		data=[
			{
				"vector": embeddings.embed_query("Query unsuccessful"),
				"text": "Query unsuccessful",
				"id": "1",
			}
		],
		mode="overwrite",
	)
	# st.write(temp_file_path)
	temp_file_path = upload_file_streamlit()
	if temp_file_path:
		loader = PyPDFLoader(temp_file_path)
		documents = loader.load_and_split()
		db = LanceDB.from_documents(documents, embeddings, connection=table)
		return db
	
# agents ,vectorstores, wiki
# https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval
# note tool
@tool("Document search")
def document_search(query: str) -> str:
	# this is the prompt to the tool itself
	"Use this function first to search for documents pertaining to the query before going into the internet"
	docs = st.session_state.vectorstore.similarity_search(query)
	docs = docs[0].page_content
	json_string = json.dumps(docs, ensure_ascii=False, indent=4)
	return json_string

# combine vector store and internet search
def ex17_agent_bot():
	st.title("🦜 LangChain: Chat with internet search")

	st.session_state.vectorstore = vectorstore_creator()

	msgs = StreamlitChatMessageHistory()
	memory = ConversationBufferMemory(
		chat_memory=msgs,
		return_messages=True,
		memory_key="chat_history",
		output_key="output",
	)
	if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
		msgs.clear()
		msgs.add_ai_message("How can I help you?")
		st.session_state.steps = {}

	avatars = {"human": "user", "ai": "assistant"}
	for idx, msg in enumerate(msgs.messages):
		with st.chat_message(avatars[msg.type]):
			# Render intermediate steps if any were saved
			for step in st.session_state.steps.get(str(idx), []):
				if step[0].tool == "_Exception":
					continue
				with st.status(
					f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
				):
					st.write(step[0].log)
					st.write(step[1])
			st.write(msg.content)

	if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
		st.chat_message("user").write(prompt)

		llm = ChatOpenAI(
			model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, streaming=True
		)
		tools = [document_search, DuckDuckGoSearchRun(name="Internet Search")]
		chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
		executor = AgentExecutor.from_agent_and_tools(
			agent=chat_agent,
			tools=tools,
			memory=memory,
			return_intermediate_steps=True,
			handle_parsing_errors=True,
		)
		with st.chat_message("assistant"):
			st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
			response = executor(prompt, callbacks=[st_cb])
			st.write(response["output"])
			st.session_state.steps[str(len(msgs.messages) - 1)] = response[
				"intermediate_steps"
			]
