from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="/Users/grantwilkins/Downloads/ggml-model-q4_0.gguf",
    temperature=0.5,
    top_p=1,
    n_ctx=6000,
    callback_manager=callback_manager,
    verbose=True,
    device="cpu",  # Change this to "cuda" if you have a GPU
)

question = "What is the integral of sin(x^2)"
answer = llm(question)
print(answer)
