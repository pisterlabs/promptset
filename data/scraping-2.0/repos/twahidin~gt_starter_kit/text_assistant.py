#No need database
import streamlit as st
import streamlit_antd_components as sac
import tempfile
from langchain.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferWindowMemory
from main_bot import insert_into_data_table
from users_module import vectorstore_selection_interface
import openai
import datetime
import configparser
import os
import ast
config = configparser.ConfigParser()
config.read('config.ini')
NEW_PLAN  = config['constants']['NEW_PLAN']
FEEDBACK_PLAN = config['constants']['FEEDBACK_PLAN']
PERSONAL_PROMPT = config['constants']['PERSONAL_PROMPT']
DEFAULT_TEXT = config['constants']['DEFAULT_TEXT']
SUBJECTS_LIST = config.get('menu_lists','SUBJECTS_SINGAPORE')
SUBJECTS_SINGAPORE = ast.literal_eval(SUBJECTS_LIST )
GENERATE = "Lesson Generator"
FEEDBACK = "Lesson Feedback"

if "api_key" not in st.session_state:
	st.session_state.api_key = False
	if st.secrets["openai_key"] != "None":
		st.session_state.api_key  = st.secrets["openai_key"]
		os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

# Create or check for the 'database' directory in the current working directory
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)

if st.secrets["sql_ext_path"] == "None":
	WORKING_DATABASE= os.path.join(WORKING_DIRECTORY , st.secrets["default_db"])
else:
	WORKING_DATABASE= st.secrets["sql_ext_path"]


#direct load into form 
def upload_lesson_plan():

	def get_file_extension(file_name):
		return os.path.splitext(file_name)[1]

	# Streamlit file uploader to accept file input
	uploaded_file = st.file_uploader("Upload a file", type=['docx', 'txt', 'pdf'])

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

		# Process the temporary file using UnstructuredFileLoader (or any other method you need)
		#st.write(temp_file_path)
		loader = UnstructuredFileLoader(temp_file_path)
		docs = loader.load()

		st.success("File processed and added to form")

		# Removing the temporary file after processing
		os.remove(temp_file_path)
		return docs


def text_feedback():
	st.subheader("1. Written feedback ")
	function = st.text_input("Function")
	process = st.text_input("Process")
	
	
	st.subheader("2. Text upload or key in manually")
	text_content = upload_lesson_plan()
	text_feedback = st.text_area("Please provide your text either upload or type into this text box", height=500, value=text_content)
	
	st.subheader("3. Specific questions that I would like feedback on")
	feedback = st.text_area("Include specific information from your text that you want feedback on.")
	
	build = sac.buttons([
				dict(label='Feedback', icon='check-circle-fill', color = 'green'),
				dict(label='Cancel', icon='x-circle-fill', color='red'),
			], label=None, index=1, format_func='title', align='center', position='top', size='default', direction='horizontal', shape='round', type='default', compact=False)
	vectorstore_selection_interface(st.session_state.user['id'])
	if build != 'Cancel':
		feedback_template = f"""Imagine you are an experienced manager or supervisor. I'd like feedback on the text I've uploaded:
			Function: {function}
			Process: {process}
			Text Content: {text_feedback}
			Specific Feedback Areas: {feedback}
			Please provide feedback to enhance this text.
			"""
		st.success("Your text has been submitted for feedback!")
		return feedback_template

	return False

def text_generator():
	st.subheader("1. Generate a written text ")
	function = st.text_input("Enter the Function")
	process = st.text_input("Enter the Process")

	st.subheader("2. Describe the text you want to generate")
	address = st.text_input("Text is address to:")
	type_of_text = st.selectbox("Type of text", ["Email", "Letter", "Report", "Memo", "Minutes", "Proposal", "Other"])
	if type_of_text == "Other":
		type_of_text = st.text_input("Enter the type of text")
	tone = st.selectbox("Tone of text", ["Formal", "Informal"])
	
	
	st.subheader("3. Describe the content of the text")
	text_content = st.text_area("Please provide the details  of the text", height=500)
	vectorstore_selection_interface(st.session_state.user['id'])
	
	build = sac.buttons([
				dict(label='Generate', icon='check-circle-fill', color = 'green'),
				dict(label='Cancel', icon='x-circle-fill', color='red'),
			], label=None, index=1, format_func='title', align='center', position='top', size='default', direction='horizontal', shape='round', type='default', compact=False)

	if build != 'Cancel':
		generate_template = f"""Imagine you are an experienced writer, I would like you to generate a text for me:
			Function: {function}
			Process: {process}
			It is address to : {address}
			Type of text: {type_of_text}
			Tone of text: {tone}
			Content of text: {text_content}
			"""
		st.success("Your request is being processed!")
		return generate_template

	return False

#chat completion memory for streamlit using memory buffer
def template_prompt(prompt, prompt_template):
	response = openai.ChatCompletion.create(
		model=st.session_state.openai_model,
		messages=[
			{"role": "system", "content":prompt_template},
			{"role": "user", "content": prompt},
		],
		temperature=st.session_state.temp, #settings option
		stream=True #settings option
	)
	return response


def lesson_bot(prompt, prompt_template):
	try:
		if prompt:
			if "memory" not in st.session_state:
				st.session_state.memory = ConversationBufferWindowMemory(k=5)
			st.session_state.msg.append({"role": "user", "content": prompt})
			message_placeholder = st.empty()
			#check if there is any knowledge base
			if st.session_state.vs:
				docs = st.session_state.vs.similarity_search(prompt)
				resources = docs[0].page_content
				reference_prompt = f"""You may refer to this resources to improve or generate the text
										{resources}
									"""
			else:
				st.warning("There is no knowledge base linked to this generative AI form")
				reference_prompt = ""
			full_response = ""
			for response in template_prompt(prompt, reference_prompt + prompt_template):
				full_response += response.choices[0].delta.get("content", "")
				message_placeholder.markdown(full_response + "▌")
			message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state["memory"].save_context({"input": prompt},{"output": full_response})
			 # Insert data into the table
			now = datetime.now() # Using ISO format for date
			num_tokens = len(full_response + prompt)*1.3
			#st.write(num_tokens)
			insert_into_data_table(now.strftime("%d/%m/%Y %H:%M:%S"),  full_response, prompt, num_tokens)
	except Exception as e:
		st.error(e)


