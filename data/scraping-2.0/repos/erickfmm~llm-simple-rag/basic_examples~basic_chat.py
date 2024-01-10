from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate


if __name__ == "__main__":
    print("Hola mundo")
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = LlamaCpp(
        model_path="/usr/src/app/models/luna-ai-llama2-uncensored.Q2_K.gguf",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    while True:
        question = input("Ingrese su prompt: ")
        prompt = f"""
        USER: {question} #Las bases curriculares de 6to básico en Chile
        ASSISTANT:
        """
        print(llm(prompt))