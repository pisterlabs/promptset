#%%

from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.deeplake import DeepLake
from langchain.chains import RetrievalQA
import textwrap
import arxiv
import torch

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

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})

def construct_chain(topic, max_res = 10):
    
    response = arxiv.Search(topic,
                            max_results= max_res,
                            sort_order=arxiv.SortOrder.Ascending)
    
    pdf = []
    for result in response.results():
        loader = PyMuPDFLoader(file_path=result.pdf_url)
        load = loader.load()
        
        for document in load:
            document.metadata["result"] = result.entry_id
            document.metadata["file_path"] = result.pdf_url
            document.metadata["title"] = result.title
            pdf.append(document)
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=200)
    texts = text_splitter.split_documents(pdf)
    
    del pdf
            
    embeddings = instructor_embeddings
    
    db = DeepLake.from_documents(documents=texts,
                                embedding=embeddings)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        return_source_documents=True)
    return qa_chain


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text