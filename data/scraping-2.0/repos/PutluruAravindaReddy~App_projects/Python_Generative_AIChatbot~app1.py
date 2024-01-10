# pip install langchain
# pip install openai
# pip install gradio
# pip install huggingface_hub

import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from huggingface_hub import HfApi


class PromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

class ConversationBufferMemory:
    def __init__(self, memory_key):
        self.memory_key = memory_key

# Define the template for conversation prompts
template = """You are a helpful assistant to answer user queries.
{chat_history}
User: {user_message}
Chatbot:"""

# Create a PromptTemplate object
prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

# Create a ConversationBufferMemory object
memory = ConversationBufferMemory(memory_key="chat_history")
# from langchain.llms import HuggingFacePipeline
# hf = HuggingFacePipeline.from_model_id(
#     model_id="gpt2",
#     task="text-generation",)

# Define the LLMChain instance with ChatOpenAI model, prompt template, and memory object
llm_chain = LLMChain(
    llm=ChatOpenAI(temperature='0.5', model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# Function to get text response based on user input and conversation history
def get_text_response(user_message, history):
    response = llm_chain.predict(user_message=user_message, chat_history=history)
    return response

# Define the examples for the chat interface
examples = ["How are you doing?", "What are your interests?", "Which places do you like to visit?"]

# Create the chat interface using gr.ChatInterface
demo = gr.ChatInterface(fn=get_text_response, inputs=["user_message", "history"], outputs="text", examples=examples)

# Launch the chat interface
demo.launch()

if __name__ == "__main__":
    demo.launch() #To create a public link, set `share=True` in `launch()`. To enable errors and logs, set `debug=True` in `launch()`.

from huggingface_hub import notebook_login

notebook_login()

HFap = HfApi()

HUGGING_FACE_REPO_ID = "<Aravind263/MyGenerativeAIChatbot>"

mkdir /content/ChatBotWithOpenAI
wget -P  /content/ChatBotWithOpenAI/ https://s3.ap-south-1.amazonaws.com/cdn1.ccbp.in/GenAI-Workshop/ChatBotWithOpenAIAndLangChain/app.py
wget -P /content/ChatBotWithOpenAI/ https://s3.ap-south-1.amazonaws.com/cdn1.ccbp.in/GenAI-Workshop/ChatBotWithOpenAIAndLangChain/requirements.txt

cd /content/ChatBotWithOpenAI

HFap.upload_file(
    path_or_fileobj="./requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=HUGGING_FACE_REPO_ID,
    repo_type="space")

HFap.upload_file(
    path_or_fileobj="./app.py",
    path_in_repo="app.py",
    repo_id=HUGGING_FACE_REPO_ID,
    repo_type="space")