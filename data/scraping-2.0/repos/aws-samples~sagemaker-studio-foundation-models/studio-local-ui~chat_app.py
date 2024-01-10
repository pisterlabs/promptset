from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, AgentOutputParser, initialize_agent, ConversationalChatAgent, AgentExecutor
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
import streamlit as st
from sagemaker import session
import json
import re
import subprocess
from typing import Dict, List
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate


def format_messages(messages: List[Dict[str, str]]) -> List[str]:
    """
    Format messages for Llama-2 chat models.
    
    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and 
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    prompt: List[str] = []

    if messages[0]["role"] == "system":
        content = "".join(["<<SYS>>\n", messages[0]["content"], "\n<</SYS>>\n\n", messages[1]["content"]])
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]
    for user, answer in zip(messages[::2], messages[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])
    prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])
    return "".join(prompt)

f = open("endpoint_name.txt", "r")
endpoint_name = f.read()
f.close()

f = open("custom_attribute.txt", "r")
custom_attributes = f.read()
f.close()

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs):
        base_input = [{"role" : "user", "content" : prompt}]
        optz_input = format_messages(base_input)
        input_str = json.dumps({
            "inputs" : optz_input, 
            "parameters" : {**model_kwargs}
        })
        return input_str.encode('utf-8')
    
    def transform_output(self, output):
        response_json = json.loads(output.read().decode("utf-8"))
        print("response_json", response_json)
        return response_json[0]["generated_text"].removesuffix('</s>')

st.set_page_config(page_title="LangChain: Chat with Llama 2 Model", page_icon="🦜")
st.title("🦜 LangChain: Chat with Llama 2 Model")

content_handler = ContentHandler()
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="history", ai_prefix="AI Assistant"
)

llm=SagemakerEndpoint(
     endpoint_name=endpoint_name, 
     region_name=session.Session().boto_region_name, 
     model_kwargs={"max_new_tokens": 200, "top_p": 0.1, "temperature": 0.2},
     endpoint_kwargs={"CustomAttributes": custom_attributes},
     content_handler=content_handler
 )


# Add sliders for Temperature, Top p, and Top k
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.6, 0.1)
top_p = st.sidebar.slider("Top p", 0.0, 0.99, 0.9, 0.1)

# Function to reinitialize the conversation agent with new parameters
def reinitialize_agent():
    llm.model_kwargs.update({"temperature": temperature, "top_p": top_p})
    memory.clear()
    msgs.clear()  # Clear StreamlitChatMessageHistory
    st.experimental_rerun()  # Rerun the script to reset the UI

# Submit button
if st.sidebar.button("Submit"):
    reinitialize_agent()

template = """The following is a friendly conversation between a human and an AI. The AI answers a user's question briefly and is only talkative when required.

Current conversation:
{history}

Human: {input}
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

agent = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=memory
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    memory.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Please ask me a question!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.predict(input=prompt, callbacks=[st_cb])
        response = re.sub("\{.*?\}","",response)
        st.write(response)