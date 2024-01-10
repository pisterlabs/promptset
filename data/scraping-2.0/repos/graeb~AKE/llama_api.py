# from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/oskar52/code/video_transcription/llama.cpp/models/7B/ggml-model-f16.bin", 
    callback_manager=callback_manager,
    verbose=True,
    max_tokens=1024
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)
# llm = Llama(model_path="/home/oskar52/code/video_transcription/llama.cpp/models/7B/ggml-model-f16.bin")
# output = llm("Q: Name the planets in the solar system? A: ", max_tokens=512, stop=["Q:", "\n"], echo=True)
# print(output)