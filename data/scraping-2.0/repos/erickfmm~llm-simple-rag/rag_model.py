from config import config

from vectorstore import load_or_create_vectorstore
from datetime import datetime


class RAG_Model():
    def __init__(self) -> None:
        self.timestamp_created = datetime.now()
        print("Creating RAG model... at", self.timestamp_created)
        self.vectorstore = None
        self.llm = None
        #self.vectorstore = load_or_create_vectorstore()

    def rag_model_function(self, prompt1, prompt2):
        print("RAG created at...", self.timestamp_created)
        from langchain.callbacks.manager import CallbackManager
        from langchain.llms import LlamaCpp
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        
        # MY CODE:
        if self.vectorstore is None:
            self.vectorstore = load_or_create_vectorstore()
        ## END

        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        from langchain.schema import StrOutputParser

        # Prompt
        prompt = PromptTemplate.from_template(
            config["TOKEN_START"]+config["TOKEN_USER"]+prompt1+" {docs}"+config["TOKEN_ASISTANT"] #Summarize the main themes in these retrieved docs: 
        )

        # Run
        question = prompt2#"What are the approaches to Task Decomposition?"
        docs = self.vectorstore.similarity_search(question, k=2, fetch_k=5)
        
        if self.llm is None:
        #https://api.python.langchain.com/en/stable/llms/langchain.llms.llamacpp.LlamaCpp.html
            self.llm = LlamaCpp(
                model_path=config["model_filename"],
                temperature=config["model_temperature"],
                max_tokens=4098,
                n_ctx=4098,
                top_p=1,
                callback_manager=callback_manager,
                verbose=True,  # Verbose is required to pass to the callback manager
            )

        # Chain TODO: TEST
        #runnable = prompt | llm() | StrOutputParser()
        #result = runnable.invoke({"docs": docs})
        
        
        # Chain Legacy
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = llm_chain(docs)

        # Output
        print(result["text"])
        return result["text"], docs

if __name__ == "__main__":
    import sys
    from os.path import join
    import subprocess
    subprocess.run([sys.executable, join("webserver", "server.py")])