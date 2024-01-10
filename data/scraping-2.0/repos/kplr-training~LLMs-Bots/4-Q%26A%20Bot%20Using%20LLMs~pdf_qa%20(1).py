from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from constants import *
from transformers import AutoTokenizer
import torch
import os
import re

class PdfQA:
    def __init__(self, config: dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    @classmethod
    def create_instructor_xl(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={"device": device})

    @classmethod
    def create_sbert_mpnet(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})

    @classmethod
    def create_flan_t5_xxl(cls, load_in_8bit=False):
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-xxl",
            max_new_tokens=200,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_xl(cls, load_in_8bit=False):
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-xl",
            max_new_tokens=200,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    @classmethod
    def create_flan_t5_small(cls, load_in_8bit=False):
        # Local flan-t5-small for inference
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_base(cls, load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_large(cls, load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_fastchat_t5_xl(cls, load_in_8bit=False):
        return pipeline(
            task="text2text-generation",
            model = "lmsys/fastchat-t5-3b-v1.0",
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    @classmethod
    def create_falcon_instruct_small(cls, load_in_8bit=False):
        model = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_pipeline = pipeline(
                task="text-generation",
                model = model,
                tokenizer = tokenizer,
                trust_remote_code = True,
                max_new_tokens=100,
                model_kwargs={
                    "device_map": "auto", 
                    "load_in_8bit": load_in_8bit, 
                    "max_length": 512, 
                    "temperature": 0.01,
                    "torch_dtype":torch.bfloat16,
                    }
            )
        return hf_pipeline

    # ... (other class methods)

    def init_embeddings(self) -> None:
        if self.config["embedding"] == EMB_OPENAI_ADA:
            self.embedding = OpenAIEmbeddings()
        elif self.config["embedding"] == EMB_INSTRUCTOR_XL:
            if self.embedding is None:
                self.embedding = PdfQA.create_instructor_xl()
        elif self.config["embedding"] == EMB_SBERT_MPNET_BASE:
            if self.embedding is None:
                self.embedding = PdfQA.create_sbert_mpnet()
        else:
            self.embedding = None

    def init_models(self) -> None:
        load_in_8bit = self.config.get("load_in_8bit", False)
        if self.config["llm"] == LLM_OPENAI_GPT35:
            pass
        else:
            if not self.llm:
                if self.config["llm"] == LLM_FLAN_T5_SMALL or self.config["llm"] == LLM_FLAN_T5_BASE or self.config["llm"] == LLM_FLAN_T5_LARGE:
                    question_t5_template = """
                    context: {context}
                    question: {question}
                    answer: 
                    """
                    QUESTION_T5_PROMPT = PromptTemplate(
                        template=question_t5_template, input_variables=["context", "question"]
                    )
                    self.llm = PdfQA.create_flan_t5_small(load_in_8bit=load_in_8bit)
                    self.llm.combine_documents_chain.llm_chain.prompt = QUESTION_T5_PROMPT
                else:
                    self.llm = PdfQA.create_flan_t5_xl(load_in_8bit=load_in_8bit)
                self.llm.combine_documents_chain.verbose = True
                self.llm.return_source_documents = True

    def vector_db_pdf(self) -> None:
        pdf_path = self.config.get("pdf_path", None)
        persist_directory = self.config.get("persist_directory", None)
        if persist_directory and os.path.exists(persist_directory):
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        elif pdf_path and os.path.exists(pdf_path):
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
            texts = text_splitter.split_documents(texts)
            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
        else:
            raise ValueError("NO PDF found")

    def retreival_qa_chain(self):
        if not self.vectordb:
            raise ValueError("Vector database not initialized.")
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
        hf_llm = HuggingFacePipeline(pipeline=self.llm, model_id=self.config["llm"])
        self.qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=self.retriever)
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True

    def answer_query(self, question: str) -> str:
        answer_dict = self.qa({"query": question})
        answer = answer_dict["result"]
        if self.config["llm"] == LLM_FASTCHAT_T5_XL:
            answer = self._clean_fastchat_t5_output(answer)
        return answer

    def _clean_fastchat_t5_output(self, answer: str) -> str:
        answer = re.sub(r"<pad>\s+", "", answer)
        answer = re.sub(r"  ", " ", answer)
        answer = re.sub(r"\n$", "", answer)
        return answer
