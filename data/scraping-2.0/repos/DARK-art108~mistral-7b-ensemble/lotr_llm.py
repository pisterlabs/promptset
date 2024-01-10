from langchain.llms import llamacpp
from langchain.chains import RetrievalQA
import os
llms = llamacpp(
    streaming=True,
    model_path="/Users/riteshyadav/.cache/lm-studio/models/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/mixtral-8x7b-instruct-v0.1.Q2_K.gguf",
    max_tokens=1500,
    temprature=0.75,
    top_p=1,
    gpu_layers=0,
    stream=True,
    verbose=True,
    n_threads = int(os.cpu_count()/2),
    n_ctx=4096
)

#create compression_retriver_reordered, check lotr.py

qa = RetrievalQA.from_chain_type(
      llm=llms,
      chain_type="stuff",
      retriever = compression_retriever_reordered,
      return_source_documents = True
)

