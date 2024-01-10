from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "KEYOPEN"

pdf_reader = PdfReader('/Users/praveenvadlamani/Downloads/OIRAChatbot/data.pdf')

document_text = ''
for page in pdf_reader.pages:
    page_text = page.extract_text()
    if page_text:
        document_text += page_text

text_separator = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
text_chunks = text_separator.split_text(document_text)

embeddings_model = OpenAIEmbeddings()
document_search = FAISS.from_texts(text_chunks, embeddings_model)

qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")
conversation_history = []

def chatbot_response(user_input):
    global conversation_history

    if user_input.lower() == "exit":
        return "Goodbye!"

    conversation_history.append(("User:", user_input))
    conversation_text = ' '.join([f"{role} {message}" for role, message in conversation_history])
    document_results = document_search.similarity_search(conversation_text)

    response = qa_chain.run(
        input_documents=document_results,
        question=user_input + "",
        temperature=0.6,
        
    )
    conversation_history.append(("Chatbot:", response))

    return response
