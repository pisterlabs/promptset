from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def create_chatbot(file_path, chain_type, k, llm_name, api_key):
    # Load documents from a PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Create embeddings using OpenAI GPT-3.5
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Create a vector database from the documents
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # Define a retriever for similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Create a chatbot chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=api_key), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )

    return qa

api_key = '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'
llm_name = 'gpt-3.5-turbo'

# Example usage:
file_path = '/AyurAI/chatbot/intro_ayurveda.pdf'
chain_type = 'stuff'
k = 3

chatbot = create_chatbot(file_path, chain_type, k, llm_name, api_key)

# Interaction loop
chat_history = []  # Initialize an empty chat history
while True:
    user_input = input("You: ")
    
    if user_input == "exit":
        exit()
        
    # Create a dictionary with the user's question and the chat history
    input_dict = {
        "question": user_input,
        "chat_history": chat_history
    }
    
    # Pass the input dictionary to the chatbot
    response = chatbot(input_dict)
    
    # Extract and print just the answer
    answer = response.get("answer", "Chatbot: I don't know the answer to that question.")
    
    # Limit the response to a single sentence
    answer = answer.split('.')[0] + '.'

    print(answer)
    
    # Update the chat history with the user's question and the chatbot's response
    # chat_history.append(user_input)
    # chat_history.append(answer)