
import os
import sys
import time
import gen_question

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Import necessary modules
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import subprocess



db_name_to_search = "db3"

def start_ret_script(new_web):
    # Specify the Python script you want to run
    python_script = "real_time_ret.py"

    # Launch a new terminal and execute the script
    terminal_command = f"python {python_script} {new_web};"
    subprocess.Popen(terminal_command, shell=True)

def change_web(new_webiste):
    global db_name_to_search

    print("NEW", new_webiste)
    
    name = new_webiste.replace("https", "")
    name = name.replace("http", "")
    name = name.replace("://", "")
    name = name.replace("www.", "")
    name = name.split(".")[0]
    name = "db_" + name

    print(name)

    if "moveworks" in name:
        name = "db3"

    print("changed")

    with open("db_to_file.txt", "w") as file:
    # Write data to the file
        file.write(name)

    return name


def initialise_qa(db_to_use):
    # Define the directory where you want to save the persisted database
    global db_name_to_search
    print("Use", db_to_use)
    db_name_to_search = db_to_use
    persist_directory = db_name_to_search

    print(persist_directory)

    # Initialize OpenAIEmbeddings for embedding
    embedding = OpenAIEmbeddings()


    # Load the persisted database from disk
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Create a conversation buffer memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0), chain_type="stuff", retriever=vectordb.as_retriever(), memory=memory)

    return qa

input_history = []
chat_history = []

def main_output(user_input, qa):
    
    global chat_history

    global input_history
    global chat_history  # Initialize chat_history as a list

    chat_history.append({"role": "user", "content": user_input})

    # Start a new thread by default
    thread_status = '1'

    input_history.append({"question": user_input})

    # Generate questions and answers
    question = user_input
    result = qa({"question": question, "chat_history": chat_history})

    if "I don't know" in result["answer"] or "not provided in the context" in result["answer"]:
        ans_open = gen_question.ask_gen_question(question)
        chat_history.append({"role": "assistant", "content": result["answer"]})
        ans_open += "\nGenerated from the internet!"

        return ans_open

    else:
        # Add AI's response to chat history

        chat_history.append({"role": "assistant", "content": result["answer"]})

        return result["answer"]


def clear_history():
    global chat_history
    chat_history = []


# if __name__ == "__main__":
#     print(init("Summarise the text"))
