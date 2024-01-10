import importlib
import os
from typing import Any, List, Callable, Optional, Sequence

import huggingface_hub
import torch
import transformers
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel, Document
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, TextStreamer)

import configs.common as config
from configs.common import model_config

class AiDatabase:
    stopRequested = False

    def __init__(self, callbacks: 'list[BaseCallbackHandler]' = [], 
                 streamerClassType: 'type[TextStreamer]' = TextStreamer):
        """
        Args:
            callbacks: Use this to implement LangChain's CallbackHandler
            streamerClassType: Use this to implement HuggingFace's CallbackHandler
        """
        print("[+] Preparing Chroma DB")
        embedding_func = SentenceTransformerEmbeddings(model_name=model_config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
        self.chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)
        print("[+] Chroma # of collections: ", self.chromadb._collection.count())

        self.retriever = self.chromadb.as_retriever(search_kwargs={"k": config.USE_TOP_K_SIMILAR_DOC, "include_metadata": True})
        if model_config.IS_GGML:
            self.llm = createCLLM(callbacks)
        else:
            self.llm = createLLM(callbacks, streamerClassType, [self.isStopRequested])
        self.retrievalQA = createRetrievalQA(self.llm, self.retriever)

        # self.agent: 'AgentExecutor' = initialize_agent(
        #     tools=[Tool(name="Chromadb QA System", func=self.retrievalQA, description="Let your AI handle the requests")],
        #     llm=self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    def reloadPrompt(self):
        importlib.reload(model_config)
        self.retrievalQA = createRetrievalQA(self.llm, self.retriever)

    def query(self, queryStr: str) -> 'list[Document]':
        if queryStr == "": 
            return
        
        self.stopRequested = False
        import time
        t = time.perf_counter()
        res = self.retrievalQA({"query": queryStr})
        print("\n[*] Inference time elapsed: ", time.perf_counter() - t)
        return res["source_documents"]
        # print(res["result"])
        # for source in res["source_documents"]:
        #     print(source.metadata)

    def isStopRequested(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return self.stopRequested

    def stopLLM(self):
        if model_config.IS_GGML:
            self.llm.stopGen()
        else:
            self.stopRequested = True

    class Record:
        id: str
        metadata: str

    def getAllDocs(self) -> 'list[Record]':
        collection = self.chromadb._collection.get(include=["metadatas"])
        if not collection:
            return []
        
        ids = collection["ids"]
        metadatas = collection["metadatas"]
        result = []
        for i in range(len(ids)):
            result.append({"id": ids[i], "metadata": metadatas[i]})
        return result

    def addDocsToDb(self, docs: 'list[Document]'):
        self.chromadb.add_documents(docs)
    
    def deleteDocsFromDb(self, ids: 'list[str]'):
        print("[+] Deleting docs: ", ids)
        self.chromadb.delete(ids)


def createCLLM(callbacks: 'list[BaseCallbackHandler]' = [StreamingStdOutCallbackHandler()]):
    """Create C/C++ based LLM (eg. w/ GGML)
    """
    print("[+] Loading C LLM model")
    binFullPath = None
    if not os.path.exists(model_config.LLM_MODEL):
        llmModelFolder, filename = downloadOneModelFile(
            model_config.LLM_MODEL, 
            specificModelBinPattern=None if not hasattr(model_config, "LLM_MODEL_BIN_FILE") else model_config.LLM_MODEL_BIN_FILE)
        llmModelFolder = os.path.normpath(llmModelFolder)
        binFullPath = os.path.join(llmModelFolder, filename)

    gpu_layers = 0
    if not config.DEVICE == "cpu":
        gpu_layers = config.GPU_LAYERS

    if config.USE_LLAMACPP_INSTEAD_OF_CTRANSFORMERS:
        from .cancellablellm.llamacpp import CancellableLlamaCpp
        if not binFullPath:
            print("[-] A '.bin' filename is required, did you forget to specify 'LLM_MODEL_BIN_FILE' in model_config?")
        return CancellableLlamaCpp(
            streaming=True,
            verbose=False,
            model_path=binFullPath,
            callbacks=callbacks,

            top_k=40,
            top_p=0.1,
            repeat_penalty=1.176,
            temperature=0.7,
            n_batch=512,
            n_gpu_layers=gpu_layers
        )

    lib = None
    if not config.DEVICE == "cpu":
        lib = config.CTRANSFORMERS_CUDA_LIB
        print("[+] Use ctransformers lib: ", lib)

    # https://www.reddit.com/r/LocalLLaMA/comments/1343bgz/what_model_parameters_is_everyone_using/
    from .cancellablellm.ctransformers import CancellableLLM
    return CancellableLLM(
        streaming=True,
        model=llmModelFolder,
        model_type=None if not hasattr(model_config, "LLM_MODEL_TYPE") else model_config.LLM_MODEL_TYPE,
        callbacks=callbacks,
        lib=lib,
        config={
            "top_k": 40,
            "top_p": 0.1,
            "repetition_penalty": 1.176,
            "temperature": 0.7,
            "batch_size": 512,
            "gpu_layers": gpu_layers
        }
    )

def downloadOneModelFile(repoId, specificModelBinPattern = None, modelFileExt = ".bin"):
    """
    specificModelBinPattern: allows wildcard like "*.bin"
    Returns: Local folder path of repo snapshot
    """
    from huggingface_hub import HfApi, snapshot_download

    if not specificModelBinPattern:
        api = HfApi()
        repo_info = api.repo_info(repo_id=repoId, files_metadata=True)
        files = [
            (f.size, f.rfilename)
            for f in repo_info.siblings
            if f.rfilename.endswith(modelFileExt)
        ]
        if not files:
            raise ValueError(f"No model file found in repo '{repoId}'")
        filename = min(files)[1]
    else:
        filename = specificModelBinPattern

    return snapshot_download(
        repo_id=repoId,
        local_files_only=model_config.IS_LLM_LOCAL,
        cache_dir=config.CACHE_DIR,
        allow_patterns=[filename, "config.json"],
    ), filename
    

def createLLM(callbacks: 'list[BaseCallbackHandler]' = [], 
              streamerClassType: 'type[TextStreamer]' = TextStreamer, 
              stoppingCriteriaList: 'list[Callable[..., bool] | StoppingCriteria]' = []):
    """For StoppingCriteria, see
    https://stackoverflow.com/questions/68277635/how-to-implement-stopping-criteria-parameter-in-transformers-library
    """
    print("[+] Loading LLM model")
    if len(config.HF_ACCESS_TOKEN) > 0 and (not model_config.IS_GGML):
        # both use_auth
        huggingface_hub.login(token=config.HF_ACCESS_TOKEN)
        langmodel = AutoModelForCausalLM.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=model_config.IS_LLM_LOCAL, use_auth_token=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=model_config.IS_TOKENIZER_LOCAL, use_auth_token=True)
    else:
        langmodel = AutoModelForCausalLM.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=model_config.IS_LLM_LOCAL)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.LLM_MODEL, cache_dir=config.CACHE_DIR, local_files_only=model_config.IS_TOKENIZER_LOCAL)

    huggingface_hub.logout()

    streamer = streamerClassType(tokenizer, skip_prompt=config.SKIP_PROMPT)

    # https://huggingface.co/docs/transformers/main/main_classes/pipelines
    # Streaming Output
    # https://github.com/hwchase17/langchain/issues/2918
    # https://github.com/hwchase17/langchain/issues/4950
    pipeline = transformers.pipeline(
        "text-generation",
        model=langmodel,
        tokenizer=tokenizer,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList(stoppingCriteriaList),

        device_map=config.DEVICE,
        max_length=1000,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={
            "top_k": 40,
            "top_p": 0.1,
            "repetition_penalty": 1.176,
            "temperature": 0.7,
            "batch_size": 64,
        },
        # callbacks=callbacks
    )

def createRetrievalQA(llm: BaseLanguageModel, retriever: VectorStoreRetriever):
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={
            "prompt": PromptTemplate(template=model_config.PROMPT_TEMPLATE, input_variables=["context", "question"])
        },
        return_source_documents=True
    )
