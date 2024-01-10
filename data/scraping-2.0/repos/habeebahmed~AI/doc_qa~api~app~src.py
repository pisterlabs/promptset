import torch
import textwrap
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# [Urgent] Alternate Model, GPT3.5/4 or anything - if we can't switch then we would have to make a different application

# [Medium Priority] multi data source (link remote directory, file, it should be able to process that)

# Need info on - how documents process, will it process after any document is inserted into directory or #based on schedule, also will the documents need to be processed on every run or once processed they are #cached, so that we don't feeded the old documents

# [Good to have] If possible make it an application, so that we can interact through UI 

###



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
    max_length=5000,
    temperature=0,
    # top_p=0.95,
    repetition_penalty=1.15
)

local_llm = HuggingFacePipeline(pipeline=pipe)

def load_data():
    loader = DirectoryLoader(path='/home/azureuser/doc_qa/data',glob='./*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                        model_kwargs={"device": "cuda"})

    torch.cuda.empty_cache()
    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=instructor_embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True)
    return qa_chain


def process_llm_response(response):
    lines = response['result'].split('\n')
    wrapped_lines = [textwrap.fill(line, width=110) for line in lines]
    result = '\n'.join(wrapped_lines)
    source = response['source_documents'][0].metadata['source'].split('/')[-1]
    # print("\033[92mAnswer:\033[0m ",wrapped_text)
    # print("\033[92mSource:\033[0m ", response['source_documents'][0].metadata['source'].split('/')[-1])
    return result, source



if __name__ == "__main__":
    while True:
        try:
            # load_data()
            query = input("\nEnter Question Here: ")
            llm_response = qa_chain(query)
            # print(llm_response)
            process_llm_response(llm_response)
        except KeyboardInterrupt:
            break