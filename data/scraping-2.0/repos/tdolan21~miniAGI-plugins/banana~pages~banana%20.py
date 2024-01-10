import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import Banana
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
import banana_dev as client
import os

load_dotenv()

st.title("Chat with your ML using a banana server :banana:")
st.warning("This tool requires configuration of an API key and model key on Banana.")

# Create a reference to your model on Banana
my_model = client.Client(
    api_key=os.getenv("BANANA_API_KEY"),
    model_key=os.getenv("BANANA_MODEL_KEY"),
    url="https://demo-wizardlm-1-0-uncensored-llama2-13b-gptq-llg3o1csfc.run.banana.dev/",
)
question = []
# Specify the model's input JSON, what you expect 
# to receive in your Potassium app. Here is an 
# example for a basic BERT model:
inputs = {
    "prompt": question,
}

# Call your model's inference endpoint on Banana.
# If you have set up your Potassium app with a
# non-default endpoint, change the first 
# method argument ("/")to specify a 
# different route.


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())

            result, meta = my_model.call("/", inputs)
            
            st.write(result)
            st.write(meta)