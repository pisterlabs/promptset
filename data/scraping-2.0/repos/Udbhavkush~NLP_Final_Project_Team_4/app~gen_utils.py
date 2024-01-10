import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import  streamlit as st
from langchain.prompts import PromptTemplate


prompt_template = "You are an AI psychotherapist dedicated to providing emotional support and guidance to users. Your goal is to assist individuals in managing stress, understanding their emotions, and offering coping strategies for various life situations. Users can engage in conversations with you to discuss their feelings, thoughts, and concerns, and you will respond with empathy, understanding, and therapeutic insights. While you can offer support, it's important to remind users that your responses are not a substitute for professional mental health advice, and you may encourage them to seek help from qualified professionals when needed."
# generate_text = pipeline("text2text-generation", model="google/flan-t5-large")
# generate_text = pipeline(model="facebook/blenderbot-3B", torch_dtype=torch.bfloat16,
#                          trust_remote_code=True, device_map="auto", return_full_text=True)

generate_text = pipeline("text-generation", model="facebook/blenderbot-3B", decoder_no_repeat_ngram_size=3)
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

def gen_ai(context=None, prompt=None):

    if not context:
        llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
        answer= llm_context_chain.predict(instruction=f"{prompt}", context=context).lstrip()

    else:
        llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
        answer = llm_chain.predict(instruction=f"{prompt}").lstrip()

    return  answer


def side_bar():
    with st.sidebar:
        if st.button('Clear Chat'):
            # Clear the chat history
            st.session_state.messages = []