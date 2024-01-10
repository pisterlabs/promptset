from langchain.llms import OpenAI
from pdf_reader.prompt import PROMPT
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import streamlit as st
from langchain.vectorstores.pgvector import PGVector
from pdf_reader.file_handler import FileHandler
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pdf_reader.db_connection as db_connection
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import langchain
from openai import error as openai_error
from pdf_reader.Error_handling import UnsupportedFileTypeError
from io import StringIO
from langchain.chains.summarize import load_summarize_chain

# langchain.debug = True

class FileReaderApp:

	load_dotenv()

# When file is uploaded we will create a document (process_uploaded_file) and using a document we call the summarize function then using summarize we call generate_questions

	def __init__(self):
		self.collection_name = None
		self.documents = None
		self.messages = []


	def generate_questions(self,summary):
    # Load the API key from the .env file
		# load_dotenv()
		try:
			openai.api_key = os.getenv("OPENAI_API_KEY")

			# Specify the prompt to generate questions from the summary
			prompt = f"Generate only 3 complete questions from the following summary from the given token size:\n{summary}\n\nQuestions:"

			# Make an API call to OpenAI GPT-3
			response = openai.Completion.create(
				engine="text-davinci-003",
				prompt=prompt,
				max_tokens=70,
				n=1
			)
			# Extract and return the generated questions
			questions = [choice['text'] for choice in response.choices]
		except openai_error.AuthenticationError as auth_error:
			return(f"Authentication error: {str(auth_error)}")
		except openai_error.RateLimitError as rate_limit_error:
			if "Request too large" in str(rate_limit_error):
				return("token exceeded above 150000 for per request")
			else:
				return(f"please wait for 20sec and try again in question generation")
		except openai_error.APIError as api_error:
			return(f"OpenAI API error: {str(api_error)}")
		except Exception as generic_error:
			return(f"An unexpected error occurred: {str(generic_error)}")
		for ques in questions:
			return ques
	
	def summarize(self,documents):
			llm_summarize = OpenAI(temperature=0,streaming=True)
			chain = load_summarize_chain(llm_summarize, chain_type="stuff")
			summary = chain.run(documents)
			question = self.generate_questions(summary)
			return summary , question
	
			# Generate questions from the summary
			
	#need session_state to generate chat history
	def generate_chat_history(self):
		file_path = "file.txt"
		with open(file_path, 'w') as file:
			for msg in self.messages:
				if isinstance(msg, HumanMessage):
					file.write("User: " + str(msg.content) + "\n")
				else:
					file.write("Assistant: " + str(msg.content) + "\n")
		return file_path
		
	
	#need session state
	def clear_history(self):
			self.messages = []

	def process_uploaded_file(self,uploaded_file):
		try:
			
			self.collection_name = uploaded_file.name
			# Checking the file format
			if uploaded_file.content_type == "application/pdf":
				pdf_handler = FileHandler(uploaded_file)
				self.documents = pdf_handler.extract_pdf_content_to_document()
			elif uploaded_file.content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
				excel_handler = FileHandler(uploaded_file)
				self.documents = excel_handler.extract_excel_content_to_document()
			elif uploaded_file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				word_handler = FileHandler(uploaded_file)
				self.documents = word_handler.extract_word_content_to_document()
			else:
				raise UnsupportedFileTypeError(f"Unsupported file format: {uploaded_file.content_type}")
		except UnsupportedFileTypeError as e:
			return f"Error:{str(e)}"
		
		self.summarize(self.documents)
		
	def Question_Answer(self,question):
		try:
					llm = OpenAI(temperature=0)
					chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
					text_splitter = CharacterTextSplitter(
						separator="\n",
						chunk_size=1000,
						chunk_overlap=20,
						length_function=len
					)
					embeddings = OpenAIEmbeddings()
					chunks = text_splitter.split_documents(self.documents)
					db = PGVector.from_documents(
												embedding=embeddings,
												pre_delete_collection=True,
												documents=chunks,
												collection_name=self.collection_name,
												connection_string=db_connection.CONNECTION_STRING,
											)
					docs = db.similarity_search(question)
					response = chain.run(input_documents=docs, question=question)
					self.messages.append(HumanMessage(content=question))
					self.messages.append(AIMessage(content=response))
					for msg in self.messages:
						output = StringIO()
						if isinstance(msg, HumanMessage):
							output.write("User: " + str(msg.content) + "\n")
						else:           
							output.write("Assistent: " + str(msg.content) + "\n")
		except openai_error.AuthenticationError as auth_error:
			return(f"Authentication error: {str(auth_error)}")
		except openai_error.RateLimitError as rate_limit_error:
			if "Request too large" in str(rate_limit_error):
				return("token exceeded above 150000 for per request")
			else:
				return(f"please wait for 20sec and try again")
		except openai_error.APIError as api_error:
			return(f"OpenAI API error: {str(api_error)}")
		except Exception as generic_error:
			return(f"An unexpected error occurred: {str(generic_error)}")

