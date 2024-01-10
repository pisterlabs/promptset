from typing import Optional
from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain
)
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
import tempfile
from langchain.prompts import (
    PromptTemplate,
)
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import uuid
import time
import re
import os
from urllib.parse import urlencode
import requests
import threading
import json
from src.non_llm_tools.utilities import log_question_uuid_json
from langchain.tools import tool
from src.llm import LLM
llm = LLM.get_instance()

embedding = LLM.get_embedding()

BLAST_db = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/BLAST_db_faiss_index", embedding)

class BlastQueryRequest(BaseModel):
    url: str = Field(
        default="https://blast.ncbi.nlm.nih.gov/Blast.cgi?",
        description="ALWAYS USE DEFAULT, DO NOT CHANGE"
    )
    cmd: str = Field(
        default="Put",
        description="Command to execute, 'Put' for submitting query, 'Get' for retrieving results."
    )
    program: Optional[str] = Field(
        default="blastn",
        description="BLAST program to use, e.g., 'blastn' for nucleotide BLAST."
    )
    database: str = Field(
        default="nt",
        description="Database to search, e.g., 'nt' for nucleotide database."
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAST or blat query, make sure to always keep the entire sequence given."
    )
    format_type: Optional[str] = Field(
        default="Text",
        description="Format of the BLAST results, e.g., 'Text', 'XML'."
    )
    rid: Optional[str] = Field(
        None,
        description="Request ID for retrieving BLAST results."
    )
    other_params: Optional[dict] = Field(
        default={"email": "noah.bruderer@uib.no"},
        description="Other optional BLAST parameters, including user email."
    )
    max_hits: int = Field(
        default=15,
        description="Maximum number of hits to return in the BLAST results."
    )
    sort_by: Optional[str] = Field(
        default="score",
        description="Criterion to sort BLAST results by, e.g., 'score', 'evalue'."
    )
    megablast: str = Field(
        default="on", 
        description="Set to 'on' for human genome alignemnts"
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question."
    )
    full_url: Optional[str] = Field(
        default='TBF',
        description="Url used for the blast query"
    )

def BLAST_api_query_generator(question: str):
    """FUNCTION to write api call for any BLAST query, """
    BLAST_structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    runnable = create_structured_output_runnable(BlastQueryRequest, llm, BLAST_structured_output_prompt)
    #retrieve relevant info to question
    retrieved_docs = BLAST_db.as_retriever().get_relevant_documents(question+'if the question is not about a specific organism dont retrieve anything')
    #keep top 3 hits
    top_3_retrieved_docs = ''.join(doc.page_content for doc in retrieved_docs[:3])
    blast_call_obj = runnable.invoke({"input": f"{question} based on {top_3_retrieved_docs}"})
    blast_call_obj.question_uuid=str(uuid.uuid4())
    return blast_call_obj

def submit_blast_query(request_data: BlastQueryRequest):
    """FIRST function to be called for each BLAST query.
    It submits the structured BlastQueryRequest obj and return the RID.
    """
    data = {
        'CMD': request_data.cmd,
        'PROGRAM': request_data.program,
        'DATABASE': request_data.database,
        'QUERY': request_data.query,
        'FORMAT_TYPE': request_data.format_type,
        'MEGABLAST':request_data.megablast,
        'HITLIST_SIZE':request_data.max_hits,
    }
    # Include any other_params if provided
    if request_data.other_params:
        data.update(request_data.other_params)
    # Make the API call
    query_string = urlencode(data)
    # Combine base URL with the query string
    full_url = f"{request_data.url}?{query_string}"
    # Print the full URL
    request_data.full_url = full_url
    print("Full URL built by retriever:\n", request_data.full_url)
    response = requests.post(request_data.url, data=data)
    response.raise_for_status()
    # Extract RID from response
    match = re.search(r"RID = (\w+)", response.text)
    if match:
        return match.group(1)
    else:
        raise ValueError("RID not found in BLAST submission response.")
 
def fetch_and_save_blast_results(request_data: BlastQueryRequest, blast_query_return: str, save_path: str , 
                                 question: str, log_file_path: str, wait_time: int = 15, max_attempts: int = 10000):
    """SECOND function to be called for a BLAST query.
    Will look for the RID to fetch the data
    """
    file_name = f'BLAST_results_{request_data.question_uuid}.txt'
    log_question_uuid_json(request_data.question_uuid,question, file_name, save_path, log_file_path,request_data.full_url)        
    base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    check_status_params = {
        'CMD': 'Get',
        'FORMAT_OBJECT': 'SearchInfo',
        'RID': blast_query_return
    }
    get_results_params = {
        'CMD': 'Get',
        'FORMAT_TYPE': 'XML',
        'RID': blast_query_return
    }
    # Check the status of the BLAST job
    for attempt in range(max_attempts):
        status_response = requests.get(base_url, params=check_status_params)
        status_response.raise_for_status()
        status_text = status_response.text
        if 'Status=WAITING' in status_text:
            print(f"{request_data.question_uuid} results not ready, waiting...")
            time.sleep(wait_time)
        elif 'Status=FAILED' in status_text:
            with open(f'{save_path}{file_name}', 'w') as file:
                file.write("BLAST query FAILED.")
        elif 'Status=UNKNOWN' in status_text:
            with open(f'{save_path}{file_name}', 'w') as file:
                file.write("BLAST query expired or does not exist.")
            raise 
        elif 'Status=READY' in status_text:
            if 'ThereAreHits=yes' in status_text:
                print("{request_data.question_uuid} results are ready, retrieving and saving...")
                results_response = requests.get(base_url, params=get_results_params)
                results_response.raise_for_status()
                # Save the results to a file
                print(f'{save_path}{file_name}')
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write(results_response.text)
                print(f'Results saved in BLAST_results_{request_data.question_uuid}.txt')
                break
            else:
                with open(f'{save_path}{file_name}', 'w') as file:
                    file.write("No hits found")
                break
        else:
            print('Unknown status')
            with open(f'{save_path}{file_name}', 'w') as file:
                file.write("Unknown status")
            break 
    if attempt == max_attempts - 1:
        raise TimeoutError("Maximum attempts reached. Results may not be ready.")
    return file_name

class BLASTAnswerExtractor:
    """Extract answer from BLAST result files"""
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        BLAST_file_answer_extractor_prompt = """
        You have to answer the question:{question} as clear and short as possible manner, be factual!\n\
        For any kind of BLAST results use try to use the hit with the best identity score to answer the question, if it is not possible move to the next one. \n\
        Be clear, and if organism names are present in ANY of the result please use them in the answer, do not make up stuff and mention how relevant the found information is (based on the identity scores)
        Based on the information given here:\n\
        {context}
        """
        self.BLAST_file_answer_extractor_prompt = PromptTemplate(input_variables=["context", "question"], template=BLAST_file_answer_extractor_prompt)
    def query(self,  question: str, file_path: str, n: int) -> str:
        #we make a short of the top hits of the files
        first_n_lines = []
        with open(file_path, 'r') as file:
            for _ in range(n):
                line = file.readline()
                if not line:
                    break
                first_n_lines.append(line)
        # Create a temporary file and write the lines to it
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.writelines(first_n_lines)
            temp_file_path = temp_file.name       
        if os.path.exists(temp_file_path):
            print(temp_file_path)
            loader = TextLoader(temp_file_path)
        else:
            print(f"Temporary file not found: {temp_file_path}")   
        # loader = TextLoader(temp_file_path)
        documents = loader.load()
        os.remove(temp_file_path)
        #split
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        #embed
        doc_embeddings = FAISS.from_documents(docs, embedding)
        BLAST_answer_extraction_chain= ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=doc_embeddings.as_retriever(), 
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.BLAST_file_answer_extractor_prompt},
            verbose=True,
        )
        BLAST_answer= BLAST_answer_extraction_chain(question)
        return BLAST_answer

def BLAST_answer(log_file_path, question, current_uuid, n_lignes: int):
    print('in Answer function:')
    with open(log_file_path, 'r') as file:
        data = json.load(file)
    print(current_uuid)
    # Access the last entry in the JSON array
    last_entry = data[-1]
    # Extract the file path
    current_file_path = last_entry['file_path']
    print('3: Extracting answer')
    answer_extractor = BLASTAnswerExtractor()
    result = answer_extractor.query(question, current_file_path, n_lignes)
    print(result)
    for entry in data:
        if entry['uuid'] == current_uuid:
            entry['answer'] = result['answer']
            break
    with open(log_file_path, 'w') as file:
        json.dump(data, file, indent=4)  
    return result

@tool
def blast_tool(question: str):
    """BLAST TOOL, use this tool if you need to blast a dna sequence on the blast data base on ncbi"""
    log_file_path='/usr/src/app/baio/data/output/BLAST/logfile.json'
    save_file_path='/usr/src/app/baio/data/output/BLAST/'
    #generate api call
    query_request = BLAST_api_query_generator(question)
    print(query_request)
    current_uuid = query_request.question_uuid  # Get the UUID of the current request
    #submit BLAST query
    rid = submit_blast_query(query_request)
    #retrieve BLAST results
    BLAST_file_name = fetch_and_save_blast_results(query_request, rid, save_file_path, question, log_file_path)
    #extract answer
    # answer_extractor = BLASTAnswerExtractor()
    result = BLAST_answer(log_file_path,question, current_uuid, 100)
    # print(result)
    # # Update the log file with the answer for the current UUID
    # print(f'CURRENT ID IS:{current_uuid}\n')
    # print(result['answer'])
    # with file_lock:
    #     with open(log_file_path, 'r') as file:
    #         data = json.load(file)
    #     for entry in data:
    #         if entry['uuid'] == current_uuid:
    #             entry['answer'] = result['answer']
    #             break
    #     with open(log_file_path, 'w') as file:
    #         json.dump(data, file, indent=4)
    return result['answer']