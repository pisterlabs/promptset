import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

# Setting environment variables for API keys
os.environ["OPENAI_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Initializing the vectorstore using Chroma and OpenAI embeddings
vectorstore = Chroma("langchain_store", OpenAIEmbeddings(), persist_directory="./data/CHROMA_DB_STABLE")

# Initializing the GPT-3.5 chat model
llm = ChatOpenAI()

# Creating a chat prompt template for the conversation
prompt = ChatPromptTemplate(
    messages=[
        # Introduction message about the chatbot and Fogify simulation platform
        SystemMessagePromptTemplate.from_template(
            "You are a chatbot that is used to talk to users about a simulation platform called Fogify. Fogify is an emulation Framework easing the modeling, deployment and experimentation of fog testbeds..."
        ),
        # Placeholder for chat history in the conversation
        MessagesPlaceholder(variable_name="chat_history"),
        # Template to display context in the conversation
        SystemMessagePromptTemplate.from_template(
            "Context: {context}"
        ),
        # Template for user's question input
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Setting up memory to store conversation history and input
memory = ConversationBufferMemory(memory_key='chat_history', input_key="question", max_length=3000, return_messages=True)

# Creating the Language Learning Model (LLM) chain for conversation flow
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False, memory=memory)

# Function to remove extra spaces from text
def remove_extra_spaces(text):
    return " ".join(text.split())

# Function to retrieve context based on user's question using the vectorstore
def get_context(question):
    docs = vectorstore.similarity_search(question)  # Find similar documents

    context = ''
    # Concatenate content of similar documents for context
    for doc in docs:
        doc.page_content = remove_extra_spaces(doc.page_content)
        context += ' ' + doc.page_content
    return context

# Function to get user input as a question
def get_question():
    question = input("\nUser: ")
    return question

# Function to process user's question and generate AI response
def ask_ai(question):
    context = get_context(question)
    dict = {"question": question, "context": context}
    response = llm_chain(dict)
    print("CHATBOT: " + response.get("text"))
    print()

# Main function initiating the chatbot interaction
def main():
    print("CHATBOT: Hello, I am the Fogify chatbot. How can I help you?")
    
    while True:
        question = get_question()  # Get user input as a question
        if(question == 'exit'):  # Check for exit command
            break
        ask_ai(question)  # Process user's question and get AI response

if __name__ == "__main__":
    main()
