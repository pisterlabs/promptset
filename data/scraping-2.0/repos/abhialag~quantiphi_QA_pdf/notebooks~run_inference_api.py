#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import shutil
import subprocess
import argparse

import torch
from flask import Flask, jsonify, request
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings,HuggingFaceEmbeddings

from run_inference import load_model,retrieval_qa_pipeline
from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma, FAISS
from werkzeug.utils import secure_filename

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME


# In[2]:


## setting up the args - these can be made runtime args
device_type='cpu'
show_sources=True
use_history=True
save_qa=True
promptTemplate_type="llama"

logging.info(f"Display Source Documents set to: {show_sources}")
print(f"Display Source Documents set to: {show_sources}")
logging.info(f"Display Use History set to: {use_history}")
print(f"Display Use History set to: {use_history}")
logging.info(f"Display promptTemplate_type set to: {promptTemplate_type}")
logging.info(f"Display Save QA set to: {save_qa}")


# In[3]:


QA = retrieval_qa_pipeline(use_history, promptTemplate_type=promptTemplate_type,device_type=device_type)


# In[4]:


app = Flask(__name__)


# In[5]:


@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    global QA
    user_prompt = request.form.get("user_prompt")
#     user_prompt = ' Magnesium has an atomic number of 12. Which of the following statements is true of a neutral magnesium atom?'
    logging.info(f'User Prompt: {user_prompt}')
    if user_prompt:
        print(f'User Prompt: {user_prompt}')
        # Get the answer from the chain
        res = QA(user_prompt)
        answer, docs = res["result"], res["source_documents"]

        prompt_response_dict = {
            "Prompt": user_prompt,
            "Answer": answer,
        }
        
        logging.info(answer)
        
        prompt_response_dict["Sources"] = []
        for document in docs:
            prompt_response_dict["Sources"].append(
                (os.path.basename(str(document.metadata["source"])), str(document.page_content))
            )

        return jsonify(prompt_response_dict), 200
    else:
        return "No user prompt received", 400


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5110, help="Port to run the API on. Defaults to 5110.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the UI on. Defaults to 127.0.0.1. "
        "Set to 0.0.0.0 to make the UI externally "
        "accessible from other devices.",
    )
    args = parser.parse_args()

    app.run(debug=False, host=args.host, port=args.port)





