import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain;
from htmlTemplates import css, bot_template, user_template;

def get_pdf_text(pdf_docs):
	text = ""
	for pdf in pdf_docs:
		pdf_reader = PdfReader(pdf)
		for page in pdf_reader.pages:
			text += page.extract_text()
	return text

def get_text_chunks(text):
	text_splitter = CharacterTextSplitter(
		separator = "\n",
		chunk_size = 1000,
		chunk_overlap = 200,
		length_function = len
	)
	chunks = text_splitter.split_text(text)
	return chunks

def get_vectorstore(text_chunks):
	embeddings = OpenAIEmbeddings()
	#embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
	vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
	return vectorstore

def get_conversation_chain(vectorstore):
	llm = ChatOpenAI()
	memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
	conversation_chain = ConversationalRetrievalChain.from_llm(
		llm = llm,
		retriever = vectorstore.as_retriever(),
		memory = memory
	)
	return conversation_chain

def handle_userinput(user_question):
	response = st.session_state.conversation({'question': user_question})
	st.session_state.chat_history = response['chat_history']

	for i, message in enumerate(st.session_state.chat_history):
		if i % 2 == 0:
			st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
		else:
			st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)


def main():
	os.environ['OPENAI_API_KEY'] = "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"
	st.set_page_config(page_title="Attorney AI", page_icon= "📄")
	st.write(css, unsafe_allow_html = True)
	
	if "conversation" not in st.session_state:
		st.session_state.conversation = None
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = None

	st.header("Document AI 📄")
	user_question = st.text_input("Ask a question....")
	if user_question:
		handle_userinput(user_question)
 
	with st.sidebar:
		st.subheader("Company Documents")
		pdf_docs = st.file_uploader("Upload your files here!", accept_multiple_files = True)
		if st.button("Upload"):
			with st.spinner("Processing"):
				raw_text = get_pdf_text(pdf_docs)
				text_chunks = get_text_chunks(raw_text)
				vectorstore = get_vectorstore(text_chunks)
				st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
	main()

 