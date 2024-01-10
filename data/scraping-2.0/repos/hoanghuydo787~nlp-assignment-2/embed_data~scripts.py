# 1. Create chunks of granularity 1000, 500, and per sentence, respectively
import os
# dataset_path = '../data/subsections/' # for vietnamese embeddings
dataset_path = '../data/translated_subsections/' # for english embeddings

# output_folder = '../vi_embeddings/'
output_folder = '../en_embeddings/'

os.makedirs(output_folder + '1000_chunks', exist_ok=True)
os.makedirs(output_folder + '500_chunks', exist_ok=True)
os.makedirs(output_folder + 'sent_chunks', exist_ok=True)
chunks_1000, chunks_500, sent_chunks = output_folder + '1000_chunks/', output_folder + '500_chunks/', output_folder + 'sent_chunks/'

from langchain.embeddings import HuggingFaceEmbeddings
# model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
model_name = "BAAI/bge-large-en-v1.5"

model_kwargs={'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

emb_func = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import concurrent.futures
import numpy as np

large_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
small_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
folders = os.listdir(dataset_path)

for folder in folders:
    os.makedirs(chunks_1000 + folder, exist_ok=True)
    os.makedirs(chunks_500 + folder, exist_ok=True)
    os.makedirs(sent_chunks + folder, exist_ok=True)

    large_data, small_data, sentences = [], [], []
    for file in os.listdir(dataset_path + folder):
        data = json.load(open(dataset_path + folder + '/' + file, 'r', encoding='utf-8'))
        subsection_content = data['subsection_content']
        
        largechunks = large_splitter.split_text(subsection_content)
        smallchunks = small_splitter.split_text(subsection_content)
        sents = subsection_content.split('\n')
        
        doc_title = data['document_title']
        subsec_title =data['subsection_title'].split('_')[-1]
        largechunks = [f"document: {doc_title}\nsection: {subsec_title}\nsnippet: {chunk}" for chunk in largechunks]
        smallchunks = [f"document: {doc_title}\nsection: {subsec_title}\nsnippet: {chunk}" for chunk in smallchunks]
        sents = [f"document: {doc_title}\nsection: {subsec_title}\nsnippet: {sent}" for sent in sents]

        large_data.extend(largechunks)
        small_data.extend(smallchunks)
        sentences.extend(sents)
    
    embeddings_sents = emb_func.embed_documents(sentences)
    embeddings_large = emb_func.embed_documents(large_data)
    embeddings_small = emb_func.embed_documents(small_data)

    np.save(chunks_1000 + folder + '/embeddings.npy', embeddings_large)
    np.save(chunks_500 + folder + '/embeddings.npy', embeddings_small)
    np.save(sent_chunks + folder + '/embeddings.npy', embeddings_sents)

    json.dump(large_data, open(chunks_1000 + folder + '/data.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(small_data, open(chunks_500 + folder + '/data.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(sentences, open(sent_chunks + folder + '/data.json', 'w', encoding='utf-8'), ensure_ascii=False)

    

# 2. Test the newly created chunks

# load emebeddings
testfolder = "../vi_embedding/500_chunks/subsections/"
docs = json.load(open(testfolder + 'data.json', 'r', encoding='utf-8'))
embeddings = np.load(testfolder + 'embeddings.npy')


print(docs[1])

print(len(embeddings[0]))

from langchain.vectorstores import FAISS

text_embeddings = list(zip(docs, embeddings))
vectorstore = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=emb_func)

vector_retriever = vectorstore.as_retriever()

vector_retriever.get_relevant_documents("Trẻ 16-18 tháng tuổi có cần tiêm chủng không?")

print(vector_retriever)