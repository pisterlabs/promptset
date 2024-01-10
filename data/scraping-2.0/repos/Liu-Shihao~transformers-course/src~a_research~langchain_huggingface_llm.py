import torch
import transformers
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import textwrap
# pip -q install accelerate bitsandbytes
"""
LangChain + Retrieval Local LLMs for Retrieval QA - No OpenAI
https://www.youtube.com/watch?v=9ISVjh8mdlA
"""
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl",
                                              # load_in_8bit=True,
                                              # device_map='auto',
                                              #   torch_dtype=torch.float16,
                                              #   low_cpu_mem_usage=True,

                                              )
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

local_llm = HuggingFacePipeline(pipeline=pipe)
print(local_llm('What is the capital of England?'))

# Load and process the text files
loader = TextLoader('../example_data/test.txt')
# loader = DirectoryLoader('../example_data', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

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

query = "What is Flash attention?"
llm_response = qa_chain(query)
process_llm_response(llm_response)




