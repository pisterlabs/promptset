import os
import streamlit as st
import asyncio
import autogen

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent
from autogen import OpenAIWrapper
from qdrant_client import QdrantClient

# Use the local LLM server or proxy
config_list = [
    {
        "model": "anthropic.claude-v2", #the name of your running model
        "base_url": "http://localhost:8000/", #the local address of the api
    }
]

st.set_page_config(page_title="MyAssistant Playground")
st.write("""# MyAssistant Playground \n #### --powered by Bedrock Claude & AutoGen""")

## Chat
class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)

class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class ChatRetrieveAssistantAgent(RetrieveAssistantAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)

class ChatQdrantRetrieveUserProxyAgent(QdrantRetrieveUserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


# refine the prompt for Claude model to improve performance for retrieve agent
# you need to adjust PROMPT_DEFAULT and PROMPT_CODE for task "default" and "code" as well
autogen.agentchat.contrib.retrieve_user_proxy_agent.PROMPT_QA = """
You're a retrieve augmented chatbot. You answer user's questions supplied within the <question> tags based on your own knowledge and the
context provided by the user within the <context> tags.
If you can't answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.
You must give as short an answer as possible.
Context is: 
<context>
{input_context}
</context>
User's question is: 
<question>
{input_question}
</question>
"""


with st.sidebar:
    st.header("Configuration")
    name = st.text_input("Name", placeholder="Gallileo")
    instructions = st.text_area("Instructions", placeholder="You are a friendly assistant, your job is to help me answer questions about the universe.")
    selected_model = st.selectbox("Model", ['Bedrock-Claude 2', 'Bedrock-Claude 1.3', 'GPT-4 Turbo'])
    with st.container():
        st.subheader('TOOLS', divider='rainbow')
        st.button("+Add Functions")
        code_itpr = st.toggle("Code Interpreter")
        retrieval = st.toggle("Retrieval")
    with st.container():
        st.subheader('Knowledge Base', divider='rainbow')
        doc_loc = st.text_input("Location:", placeholder="Your local doc path or web URL")
        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

print(code_itpr)
if not doc_loc:
    doc_loc = "/home/ubuntu/GenAI/autogen-hijack-main/mydocs"
print("Doc location: " + doc_loc)
print(retrieval)

## upload files to local path
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    with open(os.path.join(doc_loc,uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
else:
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.divider()
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Create an event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

with st.container():
    # for message in st.session_state["messages"]:
    #    st.markdown(message)

    user_input = st.chat_input("Please input here...")

    ## Code Interpreter
    if user_input and code_itpr:
        llm_config = {
            "timeout": 600,
            "config_list": config_list
        }
        # create an AssistantAgent instance named "assistant"
        assistant = TrackableAssistantAgent(
            name="assistant", llm_config=llm_config)

        # create a UserProxyAgent instance named "user"
        user_proxy = TrackableUserProxyAgent(
                        name="user_proxy",
                        human_input_mode="NEVER",
                        max_consecutive_auto_reply=2,
                        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE") or x.get("content", "").strip() == "",
                        code_execution_config={
                            "work_dir": "coding",
                            "use_docker": False,  # set to True or image name like "python:3" to use docker
                        })

        # Define an asynchronous function
        async def initiate_chat():
            await user_proxy.a_initiate_chat(
                assistant,
                message=user_input,
            )
        
        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())


    ## Retrieval
    if user_input and retrieval:
        # 1. create an RetrieveAssistantAgent instance named "assistant"
        rag_assistant = ChatRetrieveAssistantAgent(
            name="assistant", 
            system_message="You are a helpful assistant.",
            llm_config={
                "timeout": 600,
                "config_list": config_list,
            },
        )

        # 2. create the QdrantRetrieveUserProxyAgent instance named "ragproxyagent"
        ragproxyagent = ChatQdrantRetrieveUserProxyAgent(
            name="ragproxyagent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            retrieve_config={
                "task": "qa",
                # "docs_path": "https://raw.githubusercontent.com/aws-samples/amazon-bedrock-workshop/main/README.md",
                "docs_path": doc_loc,
                "chunk_token_size": 2000,
                "model": config_list[0]["model"],
                "client": QdrantClient(":memory:"),
                "embedding_model": "BAAI/bge-small-en-v1.5",
            },
        )

        async def rag_initiate_chat():
            await ragproxyagent.a_initiate_chat(
                rag_assistant,
                problem=user_input,
            )

        # Run the asynchronous function within the event loop
        loop.run_until_complete(rag_initiate_chat())


    ## Vanilla Chatbot
    if user_input and (not code_itpr) and (not retrieval):
        print("vanilla chat.")
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # need autogen >=0.20
        client = OpenAIWrapper(config_list=config_list)
        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = client.create(
                        model="anthropic.claude-v2",
                        messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                        ]
                    )
                    placeholder = st.empty()
                    full_response = client.extract_text_or_function_call(response)[0]
                    placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
