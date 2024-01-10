import time

import chromadb
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS, EMBEDDINGS_MODEL_NAME, PERSIST_DIRECTORY, TARGET_SOURCE_CHUNKS, MODEL_TYPE, \
    MODEL_N_BATCH, MODEL_N_CTX, MODEL_PATH


class PrivateGPT:
    _instance = None
    _retrievalQA = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PrivateGPT, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def init_llm_qa(self):
        if self._retrievalQA is not None:
            return

        print('RetrievalQA is null. LLM initialization started...')

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print('embeddings initialized...')

        chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=PERSIST_DIRECTORY)
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS,
                    client=chroma_client)
        print('chromadb client initialized...')
        retriever = db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
        print('vector store retriever obtained...')

        # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = []  # if args.mute_stream else [StreamingStdOutCallbackHandler()]
        # Prepare the LLM
        match MODEL_TYPE:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=MODEL_PATH,
                               max_tokens=MODEL_N_CTX,
                               n_batch=MODEL_N_BATCH,
                               callbacks=callbacks,
                               verbose=False)
            case "GPT4All":
                llm = GPT4All(model=MODEL_PATH,
                              max_tokens=MODEL_N_CTX,
                              backend='gptj',
                              n_batch=MODEL_N_BATCH,
                              callbacks=callbacks,
                              verbose=False)
            case _default:
                # raise exception if model_type is not supported
                raise Exception(f"Model type {MODEL_TYPE} is not supported."
                                f"Please choose one of the following: LlamaCpp, GPT4All")

        self._retrievalQA = RetrievalQA.from_chain_type(llm=llm,
                                                        chain_type="stuff",
                                                        retriever=retriever,
                                                        return_source_documents=False)
        print('LLM ready!')

    def qa_prompt(self, prompt: str) -> str:
        # Check for RetrievalQA state
        if self._retrievalQA is None:
            # need to restart init_llm_qa and post init process to client
            self.init_llm_qa()

        print(f"Query prompt: {prompt}")
        # Get the answer from the chain
        start = time.time()
        res = self._retrievalQA(prompt)
        answer, docs = res['result'], []  # if args.hide_source else res['source_documents']
        end = time.time()

        time_seconds = round(end - start, 2)
        print(f"LLM reply: {answer}")
        reply = f"{answer}\n\ntime:{time_seconds}"
        return reply
