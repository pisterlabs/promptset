import os
import re
import time
import yaml
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# from langchain.agents import AgentType, Tocdol, initialize_agent
from langchain.chains import LLMChain, RetrievalQA, ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_chat import message


# Get the directory path where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to config.yaml
config_path = os.path.join(script_directory, 'config.yaml')

# Open the file using the relative path
with open(config_path) as file:
    # Your code to read the file goes here
       config = yaml.load(file, Loader=SafeLoader)

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=3, return_messages=True)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat_history = []
#pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = 'rcl'

embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

# index = pinecone.Index(index_name)
# text_key = "text"
# vector_store = Pinecone(index, embedding.embed_query, text_key)

# index = pinecone.Index(index_name)
# text_key = "text"
# vector_store = Pinecone(index, embedding.embed_query, text_key)


# Initialize Pinecone
@st.cache_resource
def start_pinecone():
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console)
    )

# Load existing Pinecone index
@st.cache_resource
def load_pinecone_existing_index():
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)
    return docsearch

# Use these functions
start_pinecone()
vector_store = load_pinecone_existing_index()



def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text


def clean_text(text):
    cleaned_text = text.strip().strip("[]")
    cleaned_text = cleaned_text.replace("\n\n", "\n")
    cleaned_text = re.sub('<.*?>', '', cleaned_text)
    return cleaned_text


def get_docs(topic):
    docs = vector_store.similarity_search(
        topic,  # our search query
        k=8  # return 4 most relevant docs
    )
    return docs


def generate_response(topic):
    docs = vector_store.similarity_search(
        topic,  # our search query
        k=12  # return 8 most relevant docs
    )

    # Remove a doc if it has already been used
    used_sources = []
    sources = []
    for doc in docs:
        cleaned_doc_content = clean_text(doc.page_content)
        if cleaned_doc_content not in used_sources:
            used_sources.append(cleaned_doc_content)
            sources.append(doc)

    context = " ".join([doc.page_content for doc in sources])

    # Load previous chat history if available, else an empty list
    memory_variables = st.session_state.buffer_memory.load_memory_variables({})
    chat_history = memory_variables.get('history', [])
    # create a single input
    input = {'topic': topic, 'context': context,
             'chat_history': chat_history}
    output = chain_summarize.apply([input])

    # Extract the text from the dictionary
    text = output[0]['text']

    formatted_text = clean_text(text)

    sources_list = []
    sources_text = ""
    source_count = 0

    for i, source in enumerate(sources):
        cleaned_source = clean_text(source.page_content)
        source_count += 1
        sources_list.append(
            f"Source {source_count}:\n{cleaned_source}\n---------------------------------------------------------------------------")
       # sources_text += f"Source {source_count}:\n{cleaned_source}\n-----------"

        # Join all the cleaned sources into one string
        sources_text = "".join(sources_list)

    a = sources_text
    result = f"{formatted_text}\n\nSources:\n`{a}`"
    st.session_state.buffer_memory.save_context(
        {"input": str(topic)}, {"output": str(formatted_text)})

    # Also save the interaction in Streamlit's session state
    st.session_state['history'] = {"input": str(
        topic), "output": str(formatted_text)}
    return result


# templates
start_template = PromptTemplate(
    template="You are an Profeshinal AI model trained on valiant solutions reuable content library, your purpose it to help writters find annd generate content for their projects using yourcustom knowledge base of the reuable content library. When you respond you will respond with bullet points with answers to the questoins asked",
    input_variables=[]
)


script_template = PromptTemplate(
    template="""
    You are an AI model trained on Valiant Solutions' reusable content library. Your purpose is to assist writers answering questions for proposals using your custom knowledge base from the reusable content library. When responding focusing on detail, clarity, precision quantiative data.

    Use the context and topic provided below to generate a high-quality, informative, and non-repeative response:

    History: {chat_history}
    Context: {context}
    Topic: {topic}

     Please write a long professional response based on the above speaking about Valiant solutions in the 3rd person and make sure to add a conclusion:
    """,
    input_variables=["chat_history", "context", "topic"]
)

script_template2 = PromptTemplate(
    template="""
    You are a chatbot trained on valiant solutions reusable content liabray, in this libray you have access to all previously written summaries. I need you to write summarize just like these in the same syle tone and include information in the same way using the information provided in the topic

    History: {chat_history}
    Sources: {context}
    Topic: {topic}
    """,
    input_variables=["chat_history", "context", "topic"]
)


finish_template = PromptTemplate(
    template="""
    You are an AI model trained on Valiant Solutions' reusable content library. Your purpose is to assist writers answering questions for proposals using your custom knowledge base from the reusable content library. When responding focusing on detail, clarity, precision quantiative data.

    Use the context and topic provided below to generate a high-quality, cohesive, informative, and non-repeative response:

    Context: {content}

    Please write a two paragraph professional response based on the above speaking about Valiant solutions in the 3rd person and make sure to add a conclusion:
    """,
    input_variables=["content"]
)


clean_template = PromptTemplate(
    template=""" template="You are tasked with taking the response you just created and makeing sure that it is in the third person of valiant solutions and sounds profeshinal. DO NOT USE personal pronouns at all. Make it sound profeshinal: Context: {context} Blog post:""",
    input_variables=["context"]
)


# tools = [
#     Tool(
#         name='Valiant Knowledge Base',
#         func=generate_response,
#         description=(
#             'use this tool when answeriing questions about valiant solutions to find info to answer the question '
#             'do not modify the result of the tool'
#         )
#     )
# ]

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    # model_name='gpt-3.5-turbo',
    temperature=0.25
)

# Prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# # Chain
# chain_qa = load_qa_chain(llm, chain_type="stuff",
#                          memory=ConversationBufferWindowMemory(k=2),
#                          prompt=QA_CHAIN_PROMPT, verbose=True)

# # Run


# start_chain = LLMChain(llm=llm, prompt=start_template, verbose=True)
# finish_chain = LLMChain(llm=llm, prompt=finish_template, verbose=True)

# chain_summarize = LLMChain(
#     llm=llm, prompt=script_template2, memory=st.session_state.buffer_memory, verbose=True)

chain_summarize = LLMChain(
    llm=llm,
    prompt=script_template2,
    memory=st.session_state['buffer_memory'],
    verbose=True)

chain_clean = LLMChain(
    llm=llm, prompt=clean_template, verbose=True)

# conversational memory
# conversational_memory = ConversationBufferWindowMemory(
#     memory_key='chat_history',
#     k=3,
#     return_messages=True
# )

# agent_chain = initialize_agent(
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     memory=memory
# )


# def get_conversation_chain(user_input):
#     llm = ChatOpenAI
#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain(
#         llm=llm,
#         retriever=get_docs(user_input),
#         memory=memory
#     )
#     return conversation_chain


def runapp():
    st.title("Valiant GPT Chatbot")

    name, authentication_status, username = authenticator.login(
        'Login', 'main')

    if authentication_status:
        # Display logout button and welcome message
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{name}*')

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Placeholder for chat history
        chat_placeholder = st.empty()

        # User input field at the bottom
        # user_input = st.text_input("Please type your question here")

        # if user_input:
        #     with st.spinner("Generating response..."):
        #         time.sleep(2)  # Simulating time taken to process
        #         response = generate_response(user_input)
        #         st.session_state['chat_history'].append(
        #             {'user': user_input, 'ai': response})

        user_input = st.text_input(
            "Please type your question here", key="user_query")
        if st.button("Submit"):
            # st.session_state.user_query = user_input
            with st.spinner("Generating response..."):

                # response = generate_response(st.session_state.user_query)
                # st.session_state['chat_history'].append({'user': st.session_state.user_query, 'ai': response})
                response = generate_response(user_input)
                st.session_state['chat_history'].append(
                     {'user': user_input, 'ai': response})

            # Display chat history in the placeholder
        st.markdown("### Chat History")
        for chat in reversed(st.session_state['chat_history']):
            message(f"**You:** {chat['user']}", is_user=True)
            message(f"**AI:** {chat['ai']}", is_user=False)

           # Check if user is authenticated
    elif authentication_status == False:
        # Display error message if login failed
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        # Ask for credentials if none provided
        st.warning('Please enter your username and password')


def main():
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(
            k=3, return_messages=True)

    runapp()


if __name__ == "__main__":
    main()
