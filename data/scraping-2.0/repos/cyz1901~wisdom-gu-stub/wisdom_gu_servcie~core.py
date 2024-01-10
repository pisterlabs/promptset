from langchain.llms import LlamaCpp, GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# # Verbose is required to pass to the callback manager

# llm = GPT4All(model="../models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=1024, backend='gptj', n_batch=model_n_batch, callback_manager=callback_manager, verbose=False)


# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"


# use new model using new langchain
def loadLLM() -> LLM:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return GPT4All(streaming=True, model="./models/ggml-gpt4all-j-v1.3-groovy.bin", max_tokens=1024, backend='gptj', n_batch=model_n_batch, callback_manager=callback_manager, verbose=False)

def sendMessage():
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
    llm_chain.run(question)