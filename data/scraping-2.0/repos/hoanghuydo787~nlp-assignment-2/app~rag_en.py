import json
import os
from operator import itemgetter
import timeit

import faiss
import numpy as np
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub, LlamaCpp, GPT4All
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import (BM25Retriever, EnsembleRetriever,
                                  ParentDocumentRetriever)
from langchain.schema import Document, format_document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)
from langchain.llms.huggingface_pipeline import HuggingFacePipeline



# prepare embedding
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
# DOCUMENT_DIR = "../data/subsections/subsections"
DOCUMENT_DIR = "../data/translated_subsections/translated_subsections"

docs = []
for filename in sorted(os.listdir(DOCUMENT_DIR)):
    filepath = os.path.join(DOCUMENT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        file_data = json.load(f)
        subs = text_splitter.split_text(file_data['subsection_content'])
        subs = [file_data['subsection_title'] + '\n' + text for text in subs]
        for i, sub in enumerate(subs):
            docs.append(Document(
                page_content=sub,
                metadata={
                    "filename": filename,
                    "filepath": filepath,
                    "document_name": file_data["document_name"],
                    "document_name_accent": file_data["document_name_accent"],
                    "document_title": file_data["document_title"],
                    "document_category": file_data["document_category"],
                    "subsection_name": file_data["subsection_name"],
                    "subsection_title": file_data["subsection_title"],
                    "chunk_id": i
                }
            ))

# model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
# model_name = "duyanhpam/simcse-model-phobert-base"
# model_name = "bkai-foundation-models/vietnamese-bi-encoder"
# model_name = "BAAI/bge-large-en-v1.5"
model_name = "../en_embedding_models/BAAI_bge-large-en-v1.5"

model_kwargs={'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

vib_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# vectorstore = FAISS.from_documents(
#     documents=docs,
#     embedding=vib_embeddings
# )
# vector_retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
# vectorstore.save_local('../en_index/1000_chunks')

vectorstore = FAISS.load_local(
    '../en_index/1000_chunks',
    # '../vi_index/500_chunks',
    vib_embeddings
)
vector_retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 1

retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],
                             weights=[0.5, 0.5]) 

# snippet = """U não là tình trạng các khối u hình thành trong sọ não, đe dọa tính mạng người bệnh. U não thường có 2 loại là lành tính và ác tính. Đâu là điểm chung của chúng?
# A. Đều là các bệnh nguy hiểm
# B. Đều là ung thư
# C. Nguyên nhân chính xác không thể xác định
# D. Xảy ra nhiều nhất ở người già"""
# snippet = """Brain tumors are masses formed within the skull, posing a threat to the patient's life. Generally, there are two types of brain tumors: benign and malignant. What is a common characteristic between them?
# A. Both are dangerous illnesses.
# B. Both are cancers.
# C. The exact primary cause cannot be determined.
# D. Occur most frequently in older people."""
# kags = bm25_retriever.get_relevant_documents(snippet)
# kags = vector_retriever.get_relevant_documents(snippet)
# kags = retriever.get_relevant_documents(snippet)
# print(kags)

n_gpu_layers = 1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="../models/llama-2-7b-chat.Q6_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False,
)

# llm = GPT4All(
#     model=r"C:\Users\hoang\VisualStudioCodeProjects\PythonProjects\nlp-assignment-2\models\mistral-7b-openorca.Q4_0.gguf",
#     max_tokens=2048,
# )

model = llm

# model = HuggingFacePipeline.from_model_id(
#     model_id="vilm/vietcuna-3b",
#     # model_id = "vilm/vietcuna-7b-v3",
#     # model_id = "vlsp-2023-vllm/hoa-7b",
#     # model_id = "infCapital/llama2-7b-chatvi",
#     # model_id = "NousResearch/Llama-2-7b-hf",
#     # model_id = "ura-hcmut/ura-llama-7b",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 2048},
# )

# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Do not restate question or chat history. Answer the question based on the following context and chat history:
Chat history: {chat_history}

Context: {context}

Question: {question}
"""
# template = """Trả lời câu hỏi dựa vào ngữ cảnh và lịch sử trò chuyện sau:
# Lịch sử trò chuyện: {chat_history}

# Ngữ cảnh: {context}

# Câu hỏi: {question}
# """
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# memory = ConversationBufferMemory(
#     return_messages=True, output_key="answer", input_key="question"
# )
memory = ConversationSummaryMemory(
    llm=model, return_messages=True, output_key="answer", input_key="question"
)
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# standalone_question = {
#     "standalone_question": {
#         "question": lambda x: x["question"],
#         "chat_history": lambda x: get_buffer_string(x["chat_history"]),
#     }
#     | CONDENSE_QUESTION_PROMPT
#     | model
#     | StrOutputParser(),
# }
retrieved_documents = {
    "docs": itemgetter("question") | retriever,
    "question": lambda x: x["question"],
    'chat_history': itemgetter('chat_history'),
}
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
    'chat_history': itemgetter('chat_history'),
}
answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}
# final_chain = loaded_memory | standalone_question | retrieved_documents | answer
final_chain = loaded_memory | retrieved_documents | answer
# final_chain = loaded_memory | answer
# print(final_chain)

inputs = {"question": "what is brain tumor?"}
print(inputs)
result = final_chain.invoke(inputs)
print(result['answer'] + '\n')

memory.save_context(inputs, {"answer": result["answer"]})
memory.load_memory_variables({})

inputs = {"question": "is brain tumor cureable?"}
print(inputs)
result = final_chain.invoke(inputs)
print(result['answer'] + '\n')

memory.save_context(inputs, {"answer": result["answer"]})
memory.load_memory_variables({})

inputs = {"question": "tell me stages of brain tumor?"}
print(inputs)
result = final_chain.invoke(inputs)
print(result['answer'] + '\n')

inputs = {"question": "tell me some of the symptoms of chickenpox?"}
print(inputs)
result = final_chain.invoke(inputs)
print(result['answer'] + '\n')