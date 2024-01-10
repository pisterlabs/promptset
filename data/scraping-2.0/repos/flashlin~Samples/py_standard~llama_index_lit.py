from llama_index.indices.query.base import BaseQueryEngine
from llama_index.prompts import PromptTemplate
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document, LLMPredictor, PromptHelper
from pathlib import Path
from llama_index import download_loader

import langchain_lit

# PDFReader = download_loader("PDFReader")
# loader = PDFReader()
# documents = loader.load_data(file=Path('/work/abc.pdf'))


SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)


def load_llm_model(model_name: str, callbackHandler=None):
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCPP(
        model_url=None,
        model_path=model_name,
        temperature=0.75,
        max_new_tokens=2000,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


def load_llm_model1(model_name: str):
    llm = langchain_lit.load_llm_model(model_name)
    llm_predictor = LLMPredictor(llm=llm)
    max_input_size = 1024
    num_output = 1024
    max_chunk_overlap = 0
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=1024,
        )
    return llm_predictor


def load_txt_documents(txt_path: str):
    return SimpleDirectoryReader(txt_path).load_data()


def create_vector_store_index(llm: LlamaCPP, docs: list[Document]) -> VectorStoreIndex:
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024,
                                                   embed_model="local")
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
    return index

    # query_engine = index.as_query_engine()
    # return query_engine


class ConversationalAgent:
    history = []

    def __init__(self, llm: LlamaCPP, vector_store_index: VectorStoreIndex):
        self.qa_bot = vector_store_index.as_query_engine()

    def ask(self, question: str):
        result = self.qa_bot({"question": question, "chat_history": self.history})
        self.history = [(question, result["answer"])]
        return result["answer"]