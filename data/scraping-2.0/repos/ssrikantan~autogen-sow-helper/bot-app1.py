import streamlit as st
import openai

import autogen

from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb


st.title("autogen - SOW AI Assistant")


config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": {
            "gpt-4",
            "gpt4",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-35-turbo",
            "gpt-3.5-turbo",
        }
    },
)

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])


# Accepted file formats for that can be stored in 
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

print("Accepted file formats for `docs_path`:")
print(TEXT_FORMATS)



def init_config():
    if "ragproxyagent" not in st.session_state:
        ragproxyagent = RetrieveUserProxyAgent(
            name="ragproxyagent",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            retrieve_config={
                "task": "code",
                "docs_path": "/workspaces/autogen-sow-helper/all_docs/ites/sow",  # change this to your own path, such as https://raw.githubusercontent.com/microsoft/autogen/main/README.md
                "chunk_token_size": 2000,
                "model": config_list[1]["model"],
                "client": chromadb.PersistentClient(path="/tmp/chromadb"),
                "embedding_model": "all-mpnet-base-v2",
                "get_or_create": True,  # set to True if you want to recreate the collection
            },
        )
        # ragproxyagent.human_input_mode = "ALWAYS"
        # agent_chain.run(question="hello", verbose=True,context)
        st.session_state["ragproxyagent"] = ragproxyagent

        assistant = RetrieveAssistantAgent(
            name="assistant", 
            system_message="You are a helpful assistant.",
            llm_config={
                "timeout": 600,
                "cache_seed": 42,
                "config_list": config_list,
            },
        )
        st.session_state["assistant"] = assistant
    return st.session_state["ragproxyagent"], st.session_state["assistant"]

system_prompt = "You are an AI Assistant helping with users query for SOW Creation"
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": system_prompt})

counter = 0
for message in st.session_state.messages:
    if counter == 0:
        counter += 1
        st.text_area(label="system prompt", value=system_prompt)
        continue
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

response = None
user_content_string = ''
agent_content_string = ''
if prompt := st.chat_input("How would  you like me to help with SOW Creation?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    user_proxy, assistant = init_config()
    user_proxy.initiate_chat(assistant, problem=prompt, search_string="SOW")
    autogen_response = user_proxy.chat_messages
    print("*******************", autogen_response)
    agent_contents = []
    user_contents = []

    for value_list in autogen_response.values():
        for item in value_list:
            if(item['role']) == 'user':
                # agent_contents.append(item['content'])
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown(item['content'])
            elif(item['role']) == 'assistant':
                # user_contents.append(item['content'])
                with st.chat_message("user_proxy_agent"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown(item['content'])
    
    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    #     message_placeholder.markdown(user_content_string)
        # st.session_state.messages.append({"role": "ragproxyagent", "content": response})
