import os
import json
# Import necessary modules from langchain
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from dashboard.models import PdfChat, TokenTracker
from core.calc_tokens import get_tokens


# Load environment variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# Initialize global variables
conversation_retrieval_chain = None
chat_history = []
llm = None
llm_embeddings = None
pdfChat_id = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm, llm_embeddings
    # Initialize the language model with the OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')

    llm = OpenAI(model_name="text-davinci-003", openai_api_key=api_key)
    # Initialize the embeddings for the language model
    llm_embeddings = OpenAIEmbeddings(openai_api_key=api_key)


# Function to process a PDF document
def process_document(document_path, request, name):
    global conversation_retrieval_chain, llm, llm_embeddings
    global pdfChat_id
    # Load the document

    loader = PyPDFLoader(document_path)

    documents = loader.load()
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # Create a vector store from the document chunks
    db = Chroma.from_documents(texts, llm_embeddings)
    # Create a retriever interface from the vector store
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # Save the retriever in the database so the person can came for it letter
    #pdfChat = PdfChat.objects.create(user=request.user,name=name,vector_database=json.dumps({}),answer_array="default")
    #pdfChat.save()

    #pdfChat_id = pdfChat.id
    # Create a conversational retrieval chain from the language model and the retriever
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)



# Function to process a user prompt
def process_prompt(prompt, request):
    global conversation_retrieval_chain
    global chat_history





    tokens_needed = get_tokens(prompt)
    tokens_available = TokenTracker.objects.get_or_create(user=request.user)[0]

    if int(tokens_available.token_count) > int(tokens_needed):
        new_available_tokens = int(tokens_available.token_count) - int(tokens_needed)
        tokens_available.token_count = new_available_tokens
        tokens_available.save()
        # Pass the prompt and the chat history to the conversation_retrieval_chain object
        result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
        # update the chat history
        chat_history.append((prompt, result["answer"]))
        # Return the model's response
        return result['answer']
    else:
        return "Not enough tokens buy more"


# Initialize the language model
init_llm()
