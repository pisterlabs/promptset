import streamlit as st
import openai
import sqlite3
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
import streamlit_antd_components as sac
from k_map import (
    generate_mindmap, 
    map_prompter_with_mermaid_syntax, 
    map_prompter_with_plantuml,
    generate_plantuml_mindmap,
    render_diagram
)

import configparser
import os
config = configparser.ConfigParser()
config.read('config.ini')

NEW_PLAN  = config['constants']['NEW_PLAN']
FEEDBACK_PLAN = config['constants']['FEEDBACK_PLAN']
PERSONAL_PROMPT = config['constants']['PERSONAL_PROMPT']
DEFAULT_TEXT = config['constants']['DEFAULT_TEXT']

# Create or check for the 'database' directory in the current working directory
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]

if "api_key" not in st.session_state:
	st.session_state.api_key = False
if st.secrets["openai_key"] != "None":
	st.session_state.api_key  = st.secrets["openai_key"]
	openai.api_key = st.secrets["openai_key"]
	os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]


def metacognitive_prompter(full_response):
	with st.status("Generating visuals..."):
		input = map_prompter_with_plantuml(full_response)
		uml = generate_plantuml_mindmap(input)
		image = render_diagram(uml)
		st.image(image, use_column_width=True)
		#input = map_prompter_with_mermaid_syntax(full_response)
		#generate_mindmap(input)

#response rating component	
def rating_component():
	rating_value = sac.rate(label='Response ratings:', position='left', clear=True, value=2.0, align='left', size=15, color='#25C3B0')
	return rating_value

def insert_into_data_table(date, chatbot_ans,user_prompt, tokens, value=0):
    conn = sqlite3.connect(WORKING_DATABASE)
    cursor = conn.cursor()

    # Insert data into Data_Table using preloaded session state value
    cursor.execute('''
        INSERT INTO Data_Table (date, user_id, profile_id, chatbot_ans, user_prompt, tokens, response_rating)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (date, st.session_state.data_profile["user_id"], st.session_state.data_profile["profile_id"],  chatbot_ans, user_prompt, tokens, value))

    conn.commit()
    conn.close()

#clear messages and memory
def clear_session_states():
	st.session_state.msg = []
	if "memory" not in st.session_state:
		pass
	else:
		del st.session_state["memory"]

		
#below ------------------------------ QA  base bot , K=2 memory for short term memory---------------------------------------------
#using the query from lanceDB and vector store , combine with memory
def memory_buffer_qa_component(prompt):
	#st.write(type(st.session_state.vs))
	if st.session_state.vs:
		docs = st.session_state.vs.similarity_search(prompt)
		resource = docs[0].page_content
		source = docs[0].metadata
	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=2)
	mem = st.session_state.memory.load_memory_variables({})
	#st.write(resource)
	
	prompt_template = st.session_state.chatbot + f"""
						Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
						Search Result:
						{resource}
						{source}
						History of conversation:
						{mem}
						You must quote the source of the Search Result if you are using the search result as part of the answer"""
	
	return prompt_template


#chat completion memory for streamlit using memory buffer
def chat_completion_qa_memory(prompt):
	prompt_template = memory_buffer_qa_component(prompt)
	response = openai.ChatCompletion.create(
		model=st.session_state.openai_model,
		messages=[
			{"role": "system", "content":prompt_template },
			{"role": "user", "content": prompt},
		],
		temperature=st.session_state.temp, #settings option
		presence_penalty=st.session_state.presence_penalty, #settings option
		frequency_penalty=st.session_state.frequency_penalty, #settings option
		stream=True #settings option
	)
	return response

#integration API call into streamlit chat components with memory and qa
def basebot_qa_memory():
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		if prompt := st.chat_input("What is up?"):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in chat_completion_qa_memory(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				#Response Rating
				if st.session_state.rating == True:
					feedback_value = rating_component()
				else:
					feedback_value = 0
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state["memory"].save_context({"input": prompt},{"output": full_response})
			 # Insert data into the table
			now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			#st.write(num_tokens)
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  full_response, prompt, num_tokens, feedback_value)
			if st.session_state.visuals == True:
				metacognitive_prompter(full_response)
			#metacognitive_prompter(full_response)
			
	except Exception as e:
		st.exception(e)
#below ------------------------------ base bot , K=2 memory for short term memory---------------------------------------------
#faster and more precise but no summary
def memory_buffer_component():
	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=2)
	#st.write("Messages ", messages)
	mem = st.session_state.memory.load_memory_variables({})
	#For more customisation, this can be in the config.ini file
	prompt_template = st.session_state.chatbot + f""" 
						History of conversation:
						{mem}"""
				
	return prompt_template


#chat completion memory for streamlit using memory buffer
def chat_completion_memory(prompt):
	prompt_template = memory_buffer_component()
	#st.write("Prompt Template ", prompt_template)
	response = openai.ChatCompletion.create(
		model=st.session_state.openai_model,
		messages=[
			{"role": "system", "content":prompt_template },
			{"role": "user", "content": prompt},
		],
		temperature=st.session_state.temp, #settings option
		presence_penalty=st.session_state.presence_penalty, #settings option
		frequency_penalty=st.session_state.frequency_penalty, #settings option
		stream=True #settings option
	)
	return response

#integration API call into streamlit chat components with memory
def basebot_memory():
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		if prompt := st.chat_input("What is up?"):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in chat_completion_memory(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				if st.session_state.rating == True:
					feedback_value = rating_component()
				else:
					feedback_value = 0
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state["memory"].save_context({"input": prompt},{"output": full_response})
			 # Insert data into the table
			now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			#st.write(num_tokens)
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  full_response, prompt, num_tokens, feedback_value)
			if st.session_state.visuals == True:
				metacognitive_prompter(full_response)

	except Exception as e:
		st.error(e)


#below ------------------------------ Suitable for Q & A--------------------------------------------

#below ------------------------------ base bot , no memory ---------------------------------------------
#chat completion for streamlit function
def chat_completion(prompt):
	response = openai.ChatCompletion.create(
		model=st.session_state.openai_model,
		messages=[
			{"role": "system", "content": st.session_state.chatbot},
			{"role": "user", "content": prompt},
		],
		temperature=st.session_state.temp, #settings option
		stream=True #settings option
	)
	return response

#integration API call into streamlit chat components
def basebot():
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		if prompt := st.chat_input("What is up?"):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in chat_completion(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				#Response Rating
				if st.session_state.rating == True:
					feedback_value = rating_component()
				else:
					feedback_value = 0
				now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  full_response, prompt, num_tokens, feedback_value)
			if st.session_state.visuals == True:
				metacognitive_prompter(full_response)
				
	except Exception as e:
		st.error(e)

#below ------------------------------ base bot , with vectorstore ---------------------------------------------

def qa_component(prompt):
	#st.write(type(st.session_state.vs))
	if st.session_state.vs:
		docs = st.session_state.vs.similarity_search(prompt)
		resource = docs[0].page_content
		source = docs[0].metadata
	#st.write(resource)
	
	prompt_template = st.session_state.chatbot + f"""
						Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
						Search Result:
						{resource}
						{source}
						You must quote the source of the Search Result if you are using the search result as part of the answer"""
	
	return prompt_template


#chat completion with vectorstore for streamlit 
def chat_completion_qa(prompt):
	#show the qa component results in the prompt
	prompt_template = qa_component(prompt)
	response = openai.ChatCompletion.create(
		model=st.session_state.openai_model,
		messages=[
			{"role": "system", "content":prompt_template },
			{"role": "user", "content": prompt},
		],
		temperature=st.session_state.temp, #settings option
		presence_penalty=st.session_state.presence_penalty, #settings option
		frequency_penalty=st.session_state.frequency_penalty, #settings option
		stream=True #settings option
	)
	return response

#chat completion with vectorstore for streamlit 
def basebot_qa():
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		if prompt := st.chat_input("What is up?"):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				for response in chat_completion_qa(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
				#Response Rating
				if st.session_state.rating == True:
					feedback_value = rating_component()
				else:
					feedback_value = 0
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			 # Insert data into the table
			now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			#st.write(num_tokens)
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  full_response, prompt, num_tokens, feedback_value)
			if st.session_state.visuals == True:
				metacognitive_prompter(full_response)
			#metacognitive_prompter(full_response)
			
	except Exception as e:
		st.exception(e)


#----------------------------------return search results--------------------------------------------
def return_search_raw_results(prompt):
	if st.session_state.vs:
		docs = st.session_state.vs.similarity_search(prompt)
		ans = docs[0].page_content
		source = docs[0].metadata.get('source', None)

		return f"""{ans} \n\n Source: ({source})"""

def search_bot():
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		if prompt := st.chat_input("Enter your search query"):
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				full_response = return_search_raw_results(prompt)
				message_placeholder.markdown(full_response)
				if st.session_state.rating == True:
					feedback_value = rating_component()
				else:
					feedback_value = 0
				#message_placeholder.markdown(source)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			 # Insert data into the table
			now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			#st.write(num_tokens)
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  full_response, prompt, num_tokens, feedback_value)
			if st.session_state.visuals == True:
				metacognitive_prompter(full_response)
			

	except Exception as e:
		st.error(e)









#below ------------------------------ base bot , summary memory for long conversation---------------------------------------------
#summary of conversation , requires another LLM call for every input, useful for feedback and summarising what was spoken
def memory_summary_component(prompt): #currently not in use
	if "memory" not in st.session_state:
		llm = ChatOpenAI(model_name=st.session_state.openai_model,temperature=st.session_state.temp)
		st.session_state.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)
	messages = st.session_state["memory"].chat_memory.messages
	#st.write("Messages ", messages)
	previous_summary = ""
	mem = st.session_state["memory"].predict_new_summary(messages, previous_summary)
	prompt_template = st.session_state.chatbot + f"""
						Summary of current conversation:
						{mem}"""
	
	return prompt_template 