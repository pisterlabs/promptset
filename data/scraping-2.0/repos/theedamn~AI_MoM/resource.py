from langchain import PromptTemplate, LLMChain
from huggingface_hub import hf_hub_download
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

# This file handles the prompt and GPT modules to install locally if you're using this locally you need to change the path of the model to work

template = """ You are a loving and caring mother of two children and having a Husband. Youand  your family living in India. Your son is working in USA. Your son is sending messages to you and you are seeing those messages you will get those messages in this format.Respond to his message one by one.
Son : {Question}
Keep in mind of your last conversation if it exist to continue conversation
you need to respond back to your son this format
Mother : [Response]"""


prompt = PromptTemplate(template=template, input_variables=["Question"])

hf_hub_download(repo_id="dnato/ggml-gpt4all-j-v1.3-groovy.bin", filename="ggml-gpt4all-j-v1.3-groovy.bin", local_dir="/code")
local_path= os.getcwd() + "/ggml-gpt4all-j-v1.3-groovy.bin"
llm = GPT4All(model=local_path,callbacks=[StreamingStdOutCallbackHandler()] )
llm_chain = LLMChain(prompt=prompt, llm=llm)





