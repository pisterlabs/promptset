import json
import os

import matplotlib.pyplot as plt
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.retrievers import (BM25Retriever, EnsembleRetriever,
                                  ParentDocumentRetriever)
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
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
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs={'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

vib_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

embeddings = vib_embeddings.embed_documents([doc.page_content for doc in docs])

import faiss
import numpy as np

embeddings = np.array(embeddings, dtype=np.float32)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(index)
faiss.write_index(index, '../embeddings.index')

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=vib_embeddings
)
vector_retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
vectorstore.save_local('../')

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],
                             weights=[0.5, 0.5]) 



snippet = """U não là tình trạng các khối u hình thành trong sọ não, đe dọa tính mạng người bệnh. U não thường có 2 loại là lành tính và ác tính. Đâu là điểm chung của chúng?
A. Đều là các bệnh nguy hiểm
B. Đều là ung thư
C. Nguyên nhân chính xác không thể xác định
D. Xảy ra nhiều nhất ở người già"""

kags = bm25_retriever.get_relevant_documents(snippet)

fewshot_llama = """<s>[INST] <<SYS>>
Bạn là một trợ lý thông minh chuyên giải các câu hỏi trắc nghiệm bằng context.
Hãy trả lời câu hỏi sau dựa trên context.
<<SYS>> 

### Context: Cá sấu là động vật 2 chân chuyên ăn cỏ, sống trên cạn và có tính lãnh thổ cao.

### Câu hỏi: Cá sấu là động vật gì?
    A. ăn cỏ
    B. 4 chân
    C. sống ở đầm lầy
### Trả lời: [/INST] A</s><s>[INST] ### Context: Khoa đi khám răng, bác sĩ nói răng có 4 loại, răng vàng, răng bạc, răng cỏ và răng dễ hư.

### Câu hỏi: Có bao nhiêu loại răng?
    A. 2
    B. 4
### Trả lời: [/INST] B</s><s>[INST] ### Context: {context}

### Câu hỏi: {question}
### Trả lời: [/INST]"""

import torch
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

# 1st VERSION with quantization
model_path = "vilm/vietcuna-3b"
# model_path = "vilm/vietcuna-7b-v3"
# model_path = "vlsp-2023-vllm/hoa-7b"
# model_path = "infCapital/llama2-7b-chatvi"
# model_path = "NousResearch/Llama-2-7b-hf"
# model_path = "ura-hcmut/ura-llama-7b"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # automatically store parameters on gpu, cpu or disk
            low_cpu_mem_usage=True,  # try to limit RAM
#             torch_dtype=torch.float16,  # load model in low precision to save memory
            offload_state_dict=True,  # offload onto disk if needed
            offload_folder="offload",  # offload model to `offload/`
            quantization_config=bnb_config
        )
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipel = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            batch_size=4,
#             temperature=0.001
        )

llm = HuggingFacePipeline(pipeline=pipel)

# 2nd VERSION with custom template
template = """Use the following pieces of context to answer the question at the end by choosing the correct option(s).

{context}

Question: {question}
Option: {option}
Helpful Answer:"""

qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=template,
                input_variables=["question", "context"]
            ),
        }
    )

print(qa.combine_documents_chain.llm_chain.prompt.template)

print(tokenizer.eos_token)

def process_llm_response(llm_response):
    print("Answer: ", llm_response['result'])
    print('\n\nSources:')
    for source in llm_response['source_documents']:
        print(source.metadata['document_name'])
        print(source.page_content)
        print('@'*1000)

import math

import pandas as pd

sample_query = """Tiểu Đường là nữ ca sĩ nổi tiếng ở Trung Quốc. Thời gian gần đây giọng hát của cô đi xuống trầm trọng do những triệu chứng khó chịu như đau họng, rát họng và khó nuốt. Tiểu Đường có thể đã mắc căn bệnh gì?
A. Tiểu đường
B. Chưa chắc chắn
C. Trĩ nội
"""
print(sample_query)

response = qa(sample_query)
process_llm_response(response)