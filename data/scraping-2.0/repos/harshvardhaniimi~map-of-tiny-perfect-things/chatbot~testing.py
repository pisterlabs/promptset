from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from streamlit_chat import message
import openai
import os

# read API key from text file
def read_api_key(file):
    with open(file, "r") as f:
        return f.read().strip()
    
os.environ["OPENAI_API_KEY"] = read_api_key("openai_key.txt")
openai.api_key_path = "openai_key.txt"

def find_match(input):
    """Function to find the best match for the input query from the knowledge base"""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="vectorstore", embedding_function = embeddings)
    return vectorstore.asimilarity_search(input) #default k = 4 docs

def query_refiner(conversation, query):
    """Function for refining input query based on conversation history"""
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    """Generate a string of the conversation history"""
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferMemory(return_messages=True)
            
system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are an informational chatbot named Ava that helps user learn about good cafes, restaurants and interesting places in a city. You have an embeddings based search engine to help you find the right answer. Make sure you search the database well before answering the question. If the answer is not contained within the text below, say 'I don't know' and suggest them to check Google Maps.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# this combines system message and human message templates
prompt_template = (ChatPromptTemplate
                    .from_messages([system_msg_template, 
                                    MessagesPlaceholder(variable_name="history"), 
                                    human_msg_template])
                    )

conversation = ConversationChain(memory=st.session_state.buffer_memory, 
                                 prompt=prompt_template, 
                                 llm=llm, 
                                 verbose=True)


# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# # Accept user input
# if prompt := st.chat_input("How can you help me?"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
# Long-term: Use https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("You: ", key="input")
    # query = st.chat_input(placeholder = "Type your query here...")
    if query:
        with st.spinner("Thinking..."):
            conversation_string = get_conversation_string()
            # refine query based on conversation history
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            
            # find matching two documents in knowledge base
            context = find_match(refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

# user = st.chat_message("user")
# assistant = st.chat_message("assistant")

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            # user.write(st.session_state['requests'][i])
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                # assistant.write(st.session_state['responses'][i])
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          