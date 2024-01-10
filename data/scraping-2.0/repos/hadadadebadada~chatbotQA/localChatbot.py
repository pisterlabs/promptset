
from langchain.llms import HuggingFacePipeline


import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline



###LANGCHAIN
import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

import textwrap




##local llm
# tokenizer = LlamaTokenizer.from_pretrained("junelee/wizard-vicuna-13b")

# model = LlamaForCausalLM.from_pretrained("junelee/wizard-vicuna-13b",
#                                               load_in_8bit=True,
#                                               device_map='auto',
#                                               torch_dtype=torch.float16,
#                                               low_cpu_mem_usage=True
#                                               )
                
tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

model = LlamaForCausalLM.from_pretrained("TheBloke/wizardLM-7B-HF",
                                              load_in_8bit=True,
                                              device_map='auto',
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True
                                              )
                

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

local_llm = HuggingFacePipeline(pipeline=pipe)

#print(local_llm('What is the capital of England?'))






##LANGCHAIN

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('./new_papers/openrathaus_pdf/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()
len(documents)

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## Here is the nmew embeddings being used
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)


retriever = vectordb.as_retriever(search_kwargs={"k": 3})



# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=local_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)



def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])





query = "Was ist XÖV? ONLY ANSWER IN GERMAN LANGUAGE"
llm_response = qa_chain(query)
process_llm_response(llm_response)











##############################################FALCON######################################################################




# from transformers import AutoTokenizer, AutoModelForCausalLM
# import transformers
# import torch

# model = "tiiuae/falcon-7b-instruct"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
# )
# sequences = pipeline(
#    "What is the capital of England?",
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
# %%time

# sequences = pipeline(
#    "Was ist die Hauptstadt von Deutschland?",
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")









# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip -q install sentencepiece Xformers einops
# !pip -q install langchain


# import torch
# import transformers
# from transformers import GenerationConfig, pipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import BitsAndBytesConfig
# import bitsandbytes as bnb


# # model = "tiiuae/falcon-40b"
# # model = "tiiuae/falcon-40b-instruct"
# model = "tiiuae/falcon-7b-instruct"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     device_map="auto",
# )

# %%time

# sequences = pipeline(
#    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")






# sequences = pipeline(
#    "Was ist die Hauptstadt von Deutschland?",
#     max_length=200,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")



