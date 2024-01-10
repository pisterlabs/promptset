# E. Culurciello, June 2023
# test langchain

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# llm for dialogue:
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager
llm = LlamaCpp(
    model_path="./vicuna-7b-1.1.ggmlv3.q4_0.bin", callback_manager=callback_manager, verbose=True
)

question = "What is the name of the black hole in our galaxy?"

template = """Question: {question}
Answer: be concise:"""
# Answer: Let's work this out in a step by step way to be sure we have the right answer."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain.run(question)