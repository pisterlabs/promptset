from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

def main():
    '''
    This is a simple example of using Ollama to generate text.
    You need to have local ollama installed and running.
    :return:
    '''
    MODEL_NAME = "llama2-uncensored"
    llm = Ollama(
        model=MODEL_NAME, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    llm("What is the meaning of life?")


# generate main function
if __name__ == "__main__":
    main()
