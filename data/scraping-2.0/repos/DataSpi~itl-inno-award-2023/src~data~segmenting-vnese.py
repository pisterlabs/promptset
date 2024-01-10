import pandas as pd
from langchain.document_loaders.word_document import Docx2txtLoader
# this does not work, some how, I can not install some of its requirement libs.
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
# from langchain.text_splitter import CharacterTextSplitter
import langchain.text_splitter as ts
from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
import re



loader = Docx2txtLoader(
    "../../data/raw/6. HR.03.V3.2023. Nội quy Lao động_Review by Labor Department - Final.DOCX")
doc = loader.load()[0].page_content.lower()
doc = doc.replace("\t", "")
doc = re.sub(r"\n+", r".\n\n ", doc)
replace_mapping = { # this is for replacing some unwanted characters
    "..\n": ".\n",
    ". .\n": ".\n",
    "?.\n": "?\n",
    "? .\n": "?\n",
    ":.\n": ":\n",
    ": .\n": ":\n",
    ";.\n": ",\n",
    ": .\n": ";\n"
}
for pattern, replacement in replace_mapping.items():
    doc = doc.replace(pattern, replacement)


print(doc)

# splitting into chunks
char_splt = ts.CharacterTextSplitter(separator='.', chunk_size=1000)
doc_chunks = char_splt.split_text(text=doc)


# ------ Segmenting vietnamese
# this model is for segmenting vietnamese text before tokenizing.
import py_vncorenlp
model = py_vncorenlp.VnCoreNLP(save_dir="../../models/VnCoreNLP")
# model.word_segment(doc) # this chunked the text into pieces already
segmented_chunks = []
for i in doc_chunks:
    i = model.word_segment(i)
    segmented_chunks.append(i)

segmented_chunks2=[] # now we have to rejoin each element of the list
for chunks_list in segmented_chunks:
    chunks_string="\n".join(chunks_list)
    segmented_chunks2.append(chunks_string)

# type(segmented_chunks2[0])
# def plotting_len_text_list(tex_list):
#     len_series = pd.Series([len(i) for i in tex_list])
#     len_series.plot(kind='barh', figsize=(5, len(len_series)/3))
# plotting_len_text_list(segmented_chunks2)


# ------
import torch
from transformers import AutoModel, AutoTokenizer
# from langchain.text_splitter import TextSplitter

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
# With TensorFlow 2.0+:
# from transformers import TFAutoModel
# phobert = TFAutoModel.from_pretrained("vinai/phobert-base")

# # reading the text
# file = open('../../data/processed/6. HR.03.V3.2023. Nội quy Lao động_Review by Labor Department - Final.txt', 'r')
# input_txt = file.read()
# file.close()

token_tensors=[] # this stores the token_tensor of word tokenization for each segmented text_chunk
for chunk in segmented_chunks2:
    input_ids = torch.tensor([tokenizer.encode(chunk)])
    token_tensors.append(input_ids)

# vocab_dict=tokenizer.get_vocab() # get the vocab_dict of the tokenizer
# for id in input_ids[0].tolist(): # print out the index ot the word after getting tokenized
#     for key, val in vocab_dict.items():
#         if val==id:
#             print(id, key)

features_list=[]
for tensor in token_tensors[:10]:    
    with torch.no_grad():
        features = phobert(tensor).pooler_output  # Models outputs are now tuples
        features_list.append(features)
        # I think this works, but not with text_chunks which is to big, the model will not process. 

# with torch.no_grad():
#     features = phobert(token_tensors[0])

"""___note___
features.keys()=odict_keys(['last_hidden_state', 'pooler_output'])

features.last_hidden_state is the tensor that store vectors for word-embedding.
if the input_ids has 37 tokens, features.last_hidden_state has 37 vectors length 768.

features.pooler_output is the tensor that store vectors for sentence-embedding contains
only 1 vector length 768.
"""

len(features.last_hidden_state[0])
word_vectors_list=features.last_hidden_state[0].tolist()
for index, vector in enumerate(word_vectors_list):
    print(index, len(vector))


len(features.pooler_output[0])


# # ----
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks
# chunks=get_text_chunks(text=input_txt.lower())
# for i in chunks:
#     print()
#     print(len(i), "\n", i)

# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name="vinai/phobert-base-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# vector_store = get_vectorstore(text_chunks=chunks) # this does not work
# vector_store = get_vectorstore(text_chunks=segmented_chunks2[:1]) # this does not work
