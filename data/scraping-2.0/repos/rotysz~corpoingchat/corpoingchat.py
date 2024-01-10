import sys
import os
import time
import argparse
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.vectorstores import Pinecone
import pinecone

import  fitz


def ReadAndSplitPDF(input_pdf_dir:str, chunk_size:int=8000, chunk_overlap:int=0):
    print('Processing files ....')
    all_doc_pages = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for dirpath, dirnames, filenames in os.walk(input_pdf_dir):
        for file in filenames:
            if file.endswith('.pdf'):
                print(f'Reading -> {file}')
                pymu_loader = PyMuPDFLoader(os.path.join(dirpath, file))
                pages = pymu_loader.load_and_split(text_splitter=text_splitter)
                for page in pages:
                    page.page_content = page.page_content.replace('­\n', '')
                    page.page_content = page.page_content.replace('­', '')
                all_doc_pages.extend(pages)
    return all_doc_pages

def BuildVectorDatabase(all_doc_pages, index_file_name:str, vector_store:str,sleep_time:float=0.1):
    print('Generating index ....')
    db = None
    for page_number, page in enumerate(all_doc_pages, start=1):
        print(f'Adding page {page_number}/{len(all_doc_pages)}')
        if page_number == 1:
            if vector_store == "FAISS":
                db = FAISS.from_documents([page], OpenAIEmbeddings())
            else:  # Pinecone
                db = Pinecone.from_documents([page], OpenAIEmbeddings(), index_name=index_file_name)
        else:
            db.add_documents([page])
        time.sleep(sleep_time)

    if vector_store == "FAISS":
        db.save_local(index_file_name)

    return db

def AddToVectorDatabase(new_doc_pages, index_file_name:str, vector_store:str):
    print('Adding new documents to index ....')
    if vector_store == "FAISS":
        db = FAISS.load_local(index_file_name, OpenAIEmbeddings())
    else:  # Pinecone
        db = Pinecone.from_existing_index(index_name=index_file_name, embedding=OpenAIEmbeddings())

    for page_number, page in enumerate(new_doc_pages, start=1):
        print(f'Adding page {page_number}/{len(new_doc_pages)}')
        db.add_documents([page])
        time.sleep(0.1)

    if vector_store == "FAISS":
        db.save_local(index_file_name)

    return db

def ReadIndex(input_pdf_dir:str, index_file_name:str, option:str, vector_store:str):
    if option == 'gen':
        all_doc_pages = ReadAndSplitPDF(input_pdf_dir)
        db = BuildVectorDatabase(all_doc_pages, index_file_name, vector_store)
    elif option == 'add':
        new_doc_pages = ReadAndSplitPDF(input_pdf_dir)
        db = AddToVectorDatabase(new_doc_pages, index_file_name, vector_store)
    else:
        print(f'Loading index from file .... {index_file_name}')
        if vector_store == "FAISS":
            db = FAISS.load_local(index_file_name, OpenAIEmbeddings())
        else:
            db = Pinecone.from_existing_index(index_name=index_file_name,embedding=OpenAIEmbeddings())
    return db

def GetPromptFromFile( prompt_dir, prompt_name):
    with open(os.path.join(prompt_dir, f'{prompt_name}.txt'), 'r') as file:
        return file.read()

def GetQuestion(_query, _memory, 
                _human_template, _system_template,
                _temperature=0, _max_tokens=256, _model_name='gpt-3.5-turbo-16k'):
   
    Q_PROMPT = HumanMessagePromptTemplate.from_template(_human_template)
    S_PROMPT = SystemMessagePromptTemplate.from_template(_system_template)
    chat_prompt = ChatPromptTemplate.from_messages([S_PROMPT,Q_PROMPT])
    chain = LLMChain(llm=ChatOpenAI(model_name=_model_name , temperature=_temperature, max_tokens=_max_tokens), memory=_memory, prompt=chat_prompt)
    output = chain.predict(question=_query)
    return output

def SearchMmr (vector_store, query, k, fetch_k ):
    docs_mmr = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    return docs_mmr

def SearchMultiQuery ( vector_store, model, query, k):
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs = {"k": k}), llm=model)
    
    docs = retriever_from_llm.get_relevant_documents(query=query)
    return docs


def GetAnswer(_query, _search_query, vectorstore, 
              _human_template, _system_template, 
              _temperature=0, _max_tokens=1024, _search_elements=4, 
              _model_name='gpt-3.5-turbo-16k', _searchopt='norm'):

    _search_query_int = _search_query
    if _searchopt == 'mmr':
        docs = SearchMmr(vectorstore, _search_query_int, _search_elements, _search_elements* 3)
    elif _searchopt == 'multiq':
        docs = SearchMultiQuery(vectorstore,ChatOpenAI(model_name=_model_name, temperature=_temperature, max_tokens=_max_tokens) ,
                                 _search_query_int, _search_elements)
    else:
        docs = vectorstore.similarity_search(_search_query_int, k=_search_elements)
    
    total_words = 0
    for i in range(len(docs)):
        total_words += len(docs[i].page_content.split())
        if total_words > 2500:
            docs = docs[:i]
            break
   
    H_PROMPT = HumanMessagePromptTemplate.from_template(_human_template)
    S_PROMPT = SystemMessagePromptTemplate.from_template(_system_template)

    chat_prompt = ChatPromptTemplate.from_messages([S_PROMPT,H_PROMPT])

    print(f'Pytanie -> {_query} \nSearch queries ->\n {_search_query}\n')
    chain = load_qa_chain(ChatOpenAI(model_name=_model_name, temperature=_temperature, max_tokens=_max_tokens), 
                          chain_type="stuff", prompt=chat_prompt,verbose=False)
    output = chain({"input_documents": docs, "question": _query,"search":_search_query}, return_only_outputs=False)
    return output


def PrintAnswer(output, _print_context=False):
    print(f'Odpowiedź -> {output["output_text"]}\n')

    print("Zrodła:\n")
    for doc in output["input_documents"]:
        print(f'[{len(doc.page_content.split())}, {doc.metadata["source"]} page {doc.metadata["page"]}/{doc.metadata["total_pages"]}]')
    if _print_context:
        print('Konteksty:')
        for doc in output["input_documents"]:
            print(
                f'Kontekst [{len(doc.page_content)},{len(doc.page_content.split())}, {doc.metadata}]-> {doc.page_content}\n')
    print("")
    return

def Initialize(vector_store_arg:str, index_file_path:str, option:str):
    if vector_store_arg.lower() == "pinecone":
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
            environment=os.environ["PINECONE_ENV"]  # next to api key in console
        )
        vector_store = "PINECONE"
    elif vector_store_arg.lower() == "faiss":
        vector_store = "FAISS"
    else:
        raise ValueError(f"Invalid vector_store_arg: {vector_store_arg}")

    if option not in ['gen', 'nogen', 'add']:
        raise ValueError(f"Invalid option: {option}")

    return vector_store, option

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat tool based on documents")
    parser.add_argument("input_docs", help="Input documents directory")
    parser.add_argument("index", help="Index file path name")
    parser.add_argument("option", choices=["gen", "nogen", "add"], help="Option for index generation")
    parser.add_argument("vector_store", choices=["FAISS", "pinecone"], help="Vector store to use")
    parser.add_argument("--searchopt", help="Search option for answer generation", choices=['norm', 'mmr', 'multiq'], default='norm')
    parser.add_argument("--promptdir", default="Prompts", help="Directory containing prompt templates")

    args = parser.parse_args()

    vector_store, option = Initialize(args.vector_store, args.index, args.option.lower().strip())
    
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo-16k")

    print_context = False
    db = ReadIndex(args.input_docs, args.index, option, vector_store)
    memory = ConversationBufferWindowMemory(return_messages=True,memory_key="chat_history",k=4)
    memlen=len(memory.buffer)

    while True:

        #get query from user
        query = input("Pytanie: ")
        if query.lower() == 'q':
            break
        output_q = GetQuestion(query, memory,
                               GetPromptFromFile(args.promptdir, "Question Human Template"),
                                GetPromptFromFile(args.promptdir,"Question System Template"),
                               _max_tokens=512,_model_name=model_name)
        #query = output_q

        output = GetAnswer(query,output_q, db,
                           GetPromptFromFile(args.promptdir,"Answer Human Template"),
                           GetPromptFromFile(args.promptdir,"Answer System Template"),
                           _temperature=0, _max_tokens=1024 ,
                           _search_elements=6,_model_name=model_name,_searchopt=args.searchopt)
    #    output = GetAnswer(query, query, db, _temperature=0, _max_tokens=1024, _search_elements=6)J
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(output["output_text"])
        memlen=len(memory.buffer)
        PrintAnswer(output,print_context)
    print ("Bot stopped.")