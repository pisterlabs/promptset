import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings, openai
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def setup_chain():
    # Define file path and template
    file = 'Mental_Health_FAQ.csv'
    template = """You are a language model AI developed for a mental health project. \
            You are a friendly chat buddy or virtual therapist designed to provide support and information on \
            mental health topics. Your objective is to provide accurate and empathetic responses to a wide range of \
            mental health questions, based on the 'Mental_Health_FAQ.csv' specified in file. \

            If a user's query indicates a serious mental health crisis, please suggest they seek help from a mental \
            health professional or a trusted person in their life. Remember to prioritize their well-being and safety. \

            In your responses, ensure a tone of empathy, understanding, and encouragement. Provide users with \
            resources for further reading, or self-care strategies. Keep in mind the sensitivity of the subject matter \
            and the potential vulnerability of users when crafting responses. \

            Here are some specific interaction scenarios to guide your responses:
            - If the- user asks what you can do, respond with "I'm a chat buddy or virtual therapist here to provide \
            support and information on mental health. How can I assist you?"
            - If the user starts with a greeting, respond with 'Hello! How are you doing today? How can I assist you?' \
            or something related to that
            - If a user shares their name, use it in your responses when appropriate, to cultivate a more personal and \
            comforting conversation.
            - If a user poses a mental health-related question, answer the question based on the CSV dataset. \
            If the exact question is not available, provide a response based on mental health topics.
            - If a user asks a question that is unrelated to mental health, respond with \
            'This question is out of my scope as I'm built mainly to help support you with mental health-related \
            questions. Could you please ask a question related to mental health?'

            {context}
            Question: {question}
            Answer:"""

    # Initialize embeddings, loader, and prompt
    embeddings = OpenAIEmbeddings()
    loader = CSVLoader(file_path=file, encoding='utf-8')
    docs = loader.load()
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create DocArrayInMemorySearch and retriever
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    chain_type_kwargs = {"prompt": prompt}

    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        temperature=0
    )

    # Setup RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    return chain


# Define bot avatar display function
def display_avatar():
    st.image("avatar/bot_avatar.jpeg", width=100)


# Define main function
def main():
    agent = setup_chain()

    # Set Streamlit app title and subheader
    st.title("MindMend")
    st.subheader("A Mental Wellness Support Chatbot")

    # User input text field
    user_input = st.text_input("Ask me anything! I'm here to help:")

    # Button to trigger chatbot response
    if st.button("Enter"):
        #  Get chatbot response
        response = agent.run(user_input)

        # Display bot avatar and chatbot response
        display_avatar()
        st.markdown(f"**MindMend:** {response}")


# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
