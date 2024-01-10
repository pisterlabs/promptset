import streamlit as st
from openai import OpenAI
from basecode.authenticate import return_api_key
import sqlite3
import os
import pandas as pd
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime
import streamlit as st
import openai
from basecode.authenticate import return_api_key
from langchain.tools import YouTubeSearchTool
from basecode.kb_module import display_vectorstores
from basecode.users_module import vectorstore_selection_interface
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import tool
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
import json
import uuid
import time
import io

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from getpass import getpass

cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]

# os.environ.get("OPENAI_API_KEY")
# api_key=return_api_key()
client = OpenAI(
	# defaults to os.environ.get("OPENAI_API_KEY")
	api_key=return_api_key(),
)

#Only modify the code below for the exercises and challenges


#Template 1 - form with GenAI API call
def template1_form_with_genai_call():
	st.title("Template 1 - Form with GenAI API call")
	with st.form(key='form_with_genai_call'):
		prompt = st.text_input("Enter your prompt", value="This is a test prompt.")
		submit_button = st.form_submit_button("Call the API")

		if submit_button:
			if prompt:
				openai.api_key = return_api_key()
				os.environ["OPENAI_API_KEY"] = return_api_key()
				
				MODEL = st.secrets["default_model"]
				with st.status("Calling the OpenAI API..."):
					response = client.chat.completions.create(
						model=MODEL,
						messages=[
							{"role": "system", "content": "You are a helpful assistant."},
							{"role": "user", "content": prompt},
						],
						temperature=0,
					)
					st.write(response.choices[0].message.content)
					st.write("Do something with the response here.")
			else:
				st.warning("Please enter a prompt.")

#Template 2 - Chatbot with GenAI API call
def chat_completion_stream(prompt_design, prompt):
	openai.api_key = return_api_key()
	MODEL = "gpt-3.5-turbo"
	response = client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": prompt_design},
			{"role": "user", "content": prompt},
		],
		temperature=0,  # temperature
		stream=True,  # stream option
	)
	return response

def memory_variables():
	if "memory_variables" not in st.session_state:
		st.session_state.memory_variables = {}
		st.session_state.memory_variables = ConversationBufferWindowMemory(k=3)
	memory = st.session_state.memory_variables.load_memory_variables({})
	return memory['history']

def rag_results(prompt):
	if st.session_state.vs:
		docs = st.session_state.vs.similarity_search(prompt)
		resource = docs[0].page_content
		source = docs[0].metadata
		results = "\n\nResource from knowledge base " + resource + "\n\n Source: " + source['source']
		return results
	else:
		return "No results found"

def template2_ragbot():
	# Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	st.title("Prototype Chatbot")
	left_col, right_col = st.columns(2)

	with left_col:
		# Showing Chat history
		for message in st.session_state.chat_msg:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])
		#include memory variables
		memory = memory_variables()
		memory_context = "\n\n Previous conversation" + memory
	
	with right_col:
		st.write("You can use this space with session_state to manipulate the chatbot response or provide input into the chatbot.")

	try:
		#
		if prompt := st.chat_input("What is up?"):
			# set user prompt in chat history
			st.session_state.chat_msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				# Call rag_results and pass the prompt variable into the function
				rag = rag_results(prompt)
				# streaming function
				for response in chat_completion_stream("You are a helpful assistant" + memory_context + rag, prompt):
					full_response += (response.choices[0].delta.content or "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				st.session_state.memory_variables.save_context({"input": prompt}, {"output": full_response})	
			st.session_state.chat_msg.append(
				{"role": "assistant", "content": full_response}
			)

	except Exception as e:
		st.error(e)


#template 3- Custom OpenAI Assistant with Streamlit 
def template3_openai_assistant():
	st.sidebar.title("Open AI Assistant")
	st.sidebar.divider()
	st.sidebar.markdown("Your name", unsafe_allow_html=True)
	st.sidebar.markdown("Assistant GPT")
	st.sidebar.divider()

	# Your chosen model
	MODEL = "gpt-4-1106-preview"

	# Initialize session state variables
	if "session_id" not in st.session_state:
		st.session_state.session_id = str(uuid.uuid4())

	if "run" not in st.session_state:
		st.session_state.run = {"status": None}

	if "messages" not in st.session_state:
		st.session_state.messages = []

	if "retry_error" not in st.session_state:
		st.session_state.retry_error = 0

	# File uploader for CSV, XLS, XLSX
	uploaded_file = st.file_uploader("Upload your file", type=["csv", "xls", "xlsx"])

	if uploaded_file is not None:
		# Determine the file type
		file_type = uploaded_file.type

		try:
			# Read the file into a Pandas DataFrame
			if file_type == "text/csv":
				df = pd.read_csv(uploaded_file)
			elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
				df = pd.read_excel(uploaded_file)

			# Convert DataFrame to JSON
			json_str = df.to_json(orient='records', indent=4)
			file_stream = io.BytesIO(json_str.encode())

			# Upload JSON data to OpenAI and store the file ID
			file_response = client.files.create(file=file_stream, purpose='answers')
			st.session_state.file_id = file_response.id
			st.success("File uploaded successfully to OpenAI!")

			# Optional: Display and Download JSON
			st.text_area("JSON Output", json_str, height=300)
			st.download_button(label="Download JSON", data=json_str, file_name="converted.json", mime="application/json")
		
		except Exception as e:
			st.error(f"An error occurred: {e}")
	

	# Initialize OpenAI assistant
	if "assistant" not in st.session_state:
		openai.api_key = return_api_key()
		st.session_state.assistant = openai.beta.assistants.retrieve(st.secrets["OPENAI_ASSISTANT"])
		st.session_state.thread = client.beta.threads.create(
			metadata={'session_id': st.session_state.session_id}
		)

	# Display chat messages
	elif hasattr(st.session_state.run, 'status') and st.session_state.run.status == "completed":
		st.session_state.messages = client.beta.threads.messages.list(
			thread_id=st.session_state.thread.id
		)
		for message in reversed(st.session_state.messages.data):
			if message.role in ["user", "assistant"]:
				with st.chat_message(message.role):
					for content_part in message.content:
						message_text = content_part.text.value
						st.markdown(message_text)

	# Chat input and message creation with file ID
	if prompt := st.chat_input("How can I help you?"):
		with st.chat_message('user'):
			st.write(prompt)

		message_data = {
			"thread_id": st.session_state.thread.id,
			"role": "user",
			"content": prompt
		}

		# Include file ID in the request if available
		if "file_id" in st.session_state:
			message_data["file_ids"] = [st.session_state.file_id]

		st.session_state.messages = client.beta.threads.messages.create(**message_data)

		st.session_state.run = client.beta.threads.runs.create(
			thread_id=st.session_state.thread.id,
			assistant_id=st.session_state.assistant.id,
		)
		if st.session_state.retry_error < 3:
			time.sleep(1)
			st.rerun()

	# Handle run status
	if hasattr(st.session_state.run, 'status'):
		if st.session_state.run.status == "running":
			with st.chat_message('assistant'):
				st.write("Thinking ......")
			if st.session_state.retry_error < 3:
				time.sleep(1)
				st.rerun()

		elif st.session_state.run.status == "failed":
			st.session_state.retry_error += 1
			with st.chat_message('assistant'):
				if st.session_state.retry_error < 3:
					st.write("Run failed, retrying ......")
					time.sleep(3)
					st.rerun()
				else:
					st.error("FAILED: The OpenAI API is currently processing too many requests. Please try again later ......")

		elif st.session_state.run.status != "completed":
			st.session_state.run = client.beta.threads.runs.retrieve(
				thread_id=st.session_state.thread.id,
				run_id=st.session_state.run.id,
			)
			if st.session_state.retry_error < 3:
				time.sleep(3)
				st.rerun()