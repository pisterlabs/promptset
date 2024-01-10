import os
import urllib.request
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
import streamlit as st


## We load the codellama python 7b model


def download_file(file_link, filename):
    '''
    Function to download a file on the current directory'''
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        print("Model File is curently downloading …")
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:

        print("File already exists.")

# Dowloading gguf model from HuggingFace
gguf_model_path = "https://huggingface.co/TheBloke/CodeLlama-7B-Python-GGUF/resolve/main/codellama-7b-python.Q5_K_M.gguf"
filename = "codellama-7b-python.Q5_K_M.gguf"

download_file(gguf_model_path, filename)




def generate_text(
    question='Write only: "Write something please"',
):
    ## Defining a callback --> token-wise streaming
    st_callback = StreamlitCallbackHandler(st.container())
    callback_manager = CallbackManager([st_callback])

    ## Defining the model
    llm = LlamaCpp(model_path=os.path.join(os.getcwd(), "codellama-7b-python.Q5_K_M.gguf"),    temperature=0.1,
        max_tokens=512,
        top_p=0.1,
        callback_manager=callback_manager, 
        verbose=True, # Verbose is required to pass to the callback manager
    )

    ## Defining prompt template

    template = """Question: {question}

    Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_model = LLMChain(prompt=prompt, llm=llm)

    return llm_model.run(question)
