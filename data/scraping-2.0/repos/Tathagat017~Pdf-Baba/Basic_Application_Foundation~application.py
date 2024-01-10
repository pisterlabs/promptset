from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_chatbot_responses(pdf_docs, user_question):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']
    chatbot_responses = []
    
    for i, message in enumerate(chat_history):
        if i % 2 != 0:
            chatbot_responses.append(message.content)
    
    return chatbot_responses



def main():
    load_dotenv()
    pdf_docs = ['./script.pdf'] 
    user_chat = []
    user_question = ""
    print(user_chat)
    # user=""
  
    while(True):
    
     # List of PDF file paths
        user = input("Another Question: Y/N")
        if(user=="N"):
            break;
        else:
            user_question = input("Question : ")
            chatbot_responses = get_chatbot_responses(pdf_docs, user_question)
            user_chat.append({'question': user_question, 'response': chatbot_responses[0]})
            print("Chatbot Responses:")
            for idx, response in enumerate(chatbot_responses, start=1):
                    print(f"{idx}. {response}")

if __name__ == "__main__":
    main()
