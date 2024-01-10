#importing-all-modules
# import openai 
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.vectorstores import Milvus
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
import os

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
OPENAI_API_KEY = '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'

#adding the open-ai key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# def count_tokens(chain, query):
#     with get_openai_callback() as cb:
#         result = chain.run(query)
#         print(f'Spent a total of {cb.total_tokens} tokens')

#     return result  

def clean_text(text):
    cleaned_string = text.replace("\n","").replace('..',"")
    return cleaned_string

def text_summary_generator(docs):

    llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)

    documents = docs[0].page_content[:]
    text_splitter = CharacterTextSplitter() 
    texts = text_splitter.split_text(documents)
    texts = [clean_text(text) for text in texts] #list-comprehensions
    docs = [Document(page_content=t) for t in texts[:4]]
    chain = load_summarize_chain(llm, chain_type="map_reduce",verbose = True)
    summary = chain.run(docs)
    print(summary)

def Q_and_A_ChatBot(docs):

    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    docs = text_splitter.split_documents(docs)

    # docs = [clean_text(doc) for doc in docs] #list-comprehensions

    # Set up an embedding model to covert document chunks into vector embeddings.
    embeddings = OpenAIEmbeddings(model="ada",openai_api_key= OPENAI_API_KEY)

    # print(embeddings)
    # print(type(docs[0]))

    # Set up a vector store used to save the vector embeddings. Here we use Milvus as the vector store.
    vector_store = Milvus.from_documents(
        docs,
        embedding=embeddings,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        collection_name = "Data"
    )

    # print(vector_store)

    # query = "Give the statistics of child labour in india"
    # docs = vector_store.similarity_search(query)
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
    #Creating a memory object, which is neccessary to track the inputs/outputs and hold a conversation.
    #Storing of messages and then extracts the messages in a variable.
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #Chat-over-documents-with-chat-history(it allows to pass chat history as an argument)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, memory = memory)

    chat_history = []
    while True:
        print("Do you have a query?")
        print("1: Yes ")
        print("2: No")
        choice=input()
        if choice=="1":
            print("Chat with your data. Type 'exit' to stop")
            query = input("Enter the query: ")
            if query.lower() == 'exit':
                print("Thanks for the chat!")
                return
            result = qa({"question": query, "chat_history": chat_history})
            result["answer"]
            print(result)
            chat_history.append((query, result["answer"]))
        else:
            break
    print(chat_history)

def menu_option_1():
    print("Summarization started.")

def menu_option_2():
    print("ChatBot Started.")

def menu_option_3():
    print("Exited Successfully.")

def default_option():
    print("Invalid option.")

def menuDrivenUserChoice():

    #Loading the documents
    loader = UnstructuredFileLoader("content/Documents/child_labour.txt")
    docs = loader.load()

    print("Menu:")
    print("1. Text-Summarizer")
    print("2. Enter CHATBOT!")
    print("3. Exit the terminal")
    
    choice = input("Enter your choice: ")

    # Simulating switch-case using if-elif-else statements
    if choice == "1":
        menu_option_1()
        text_summary_generator(docs)
    
    elif choice == "2":
        menu_option_2()
        Q_and_A_ChatBot(docs)

    elif choice == "3":
        menu_option_3()

    else:
        default_option()

def main():

    menuDrivenUserChoice()

if __name__ == "__main__":

    main()