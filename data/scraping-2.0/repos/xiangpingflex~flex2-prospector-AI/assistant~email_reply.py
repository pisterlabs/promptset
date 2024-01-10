import streamlit as st
from dotenv import load_dotenv

from assistant.common.constant import FINE_TUNED_GPT_4
from assistant.model.reply_model import ReplyLLM

# from assistant.common.constant import FINE_TUNED_GPT_4
# from assistant.model.reply_model import ReplyLLM
import os
import openai

load_dotenv()
openai.api_key = os.environ.get("OPEN-API-KEY")


@st.cache_data
def get_llm_model():
    print("loading model")
    return ReplyLLM(model_name=FINE_TUNED_GPT_4)


llm = get_llm_model()

# 5. Build an app with streamlit
# st.set_page_config(page_title="Customer response generator", page_icon=":bird:")
st.header("Customer response generator :bird:")
message = st.text_area("customer message")

if message:
    st.write("Generating best practice message...")
    # result = "safsad"
    result = llm.generate_reply_email(message)
    st.info(result)
