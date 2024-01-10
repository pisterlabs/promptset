
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

relative_new_training = "new_training.txt" 

relative_master = "master.txt" 

new_training_path = os.path.join(script_directory, relative_new_training)
master_path = os.path.join(script_directory, relative_master)

class DocReader():
	def __init__(self):
		load_dotenv()

	def run(self):
		st.set_page_config(page_title="Chat with multiple docs", page_icon=":books:")

		st.write(css, unsafe_allow_html=True)

		st.header("Chat with the LWFM assistant.")
		user_question = st.text_input("Ask a question about your documents:")

		if st.button("Submit"):
			self.handle_user_input(user_question)

		with st.sidebar:
			st.subheader("To reload the model training click on 'Reload'")
			if st.button("Reload"):
				with st.spinner("Processing"):
					self.reload_docs()

	def reload_docs(self):
		try:
			file_object = open(new_training_path, 'r')
			self.feed_docs([file_object])
			# Open the new training file for reading and the master file for appending
			with open(new_training_path, 'r') as new_training, open(master_path, 'a') as master:
				# Append the contents of the new training file to the master file
				master.write(new_training.read())

			# Clear the contents of the new training file by opening it in write mode
			with open(new_training_path, 'w') as new_training:
				pass  # This effectively clears the file

		except FileNotFoundError:
			print("One or both of the file paths does not exist.")
			return None

	def load_master(self):
		print("Loading Master Doc...")
		try:
			file_object = open(master_path, 'r')
			self.feed_docs([file_object])
			print("Loaded Master Doc.")

		except FileNotFoundError:
			print("The file path does not exist.")
			return None
				
	def feed_docs(self, docs):
		# get document text
		text = self.get_doc_text(docs)

		# get the text chunks
		text_chunks = self.get_text_chunks(text)

		# create vector store
		vectorstore = self.get_vectorstore(text_chunks)

		# create conversation
		st.session_state.conversation = self.get_conversation_chain(vectorstore)

	def get_doc_text(self, docs):
		combined_text = ""
		for file in docs:
			file_text = file.read()
			combined_text += file_text
		return combined_text

	def get_text_chunks(self, text):
		text_splitter = CharacterTextSplitter(
			separator="\n",
			chunk_size=1000,
			chunk_overlap=200,
			length_function=len
		)
		chunks = text_splitter.split_text(text)
		return chunks

	def get_vectorstore(self, text_chunks):
		embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
		vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
		return vectorstore

	def get_conversation_chain(self, vectorstore):
		llm = ChatOpenAI()
		memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
		conversation_chain = ConversationalRetrievalChain.from_llm(
			llm=llm,
			retriever=vectorstore.as_retriever(),
			memory=memory
		)

		return conversation_chain

	def handle_user_input(self, user_question):
		print("User Question: " + str(user_question) + "|" + str(type(user_question)))
		response = st.session_state.conversation({'question': user_question})
		st.session_state.chat_history = response['chat_history']
		for i, message in enumerate(st.session_state.chat_history):
			if i % 2 == 0:
				st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
			else:
				st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__ == '__main__':
	if "conversation" not in st.session_state:
		st.session_state.conversation = None

	if "chat_history" not in st.session_state:
		st.session_state.chat_history = None
	doc_reader = DocReader()
	doc_reader.load_master()
	doc_reader.run()
