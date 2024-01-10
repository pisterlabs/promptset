from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate


def main():
    template = """Question: {question}
    
    Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate(template, input_variables=["question"])

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])