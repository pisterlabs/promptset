import urllib.request
import urllib.parse
import json
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain
)
from langchain.vectorstores import FAISS
import os
from typing import Optional
from langchain.prompts import (
    PromptTemplate,
)
from typing import Optional, Dict
from pydantic import BaseModel, Field
from typing import Optional
import json
import uuid
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
from src.non_llm_tools.utilities import log_question_uuid_json
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from urllib.parse import urlencode
from src.llm import LLM

llm = LLM.get_instance()

embedding = LLM.get_embedding()
class BLATQueryRequest(BaseModel):
    url: str = Field(
        default="https://genome.ucsc.edu/cgi-bin/hgBlat?",
        description="For DNA alignnment to a specific genome use default"
    )
    query: Optional[str] = Field(
        None,
        description="Nucleotide or protein sequence for the BLAT query, make sure to always keep the entire sequence given."
    )
    ucsc_db: str = Field(
        default="hg38",
        description="Genome assembly to use in the UCSC Genome Browser, use the correct db for the organisms. Human:hsg38; Mouse:mm10; Dog:canFam6"
    )
    # Additional fields for UCSC Genome Browser
    ucsc_track: str = Field(
        default="genes",
        description="Genome Browser track to use, e.g., 'genes', 'gcPercent'."
    )
    ucsc_region: Optional[str] = Field(
        None,
        description="Region of interest in the genome, e.g., 'chr1:100000-200000'."
    )
    ucsc_output_format: str = Field(
        default="json",
        description="Output format for the UCSC Genome Browser, e.g., 'bed', 'fasta'."
    )
    ucsc_query_type:str = Field(
        default='DNA',
        description='depends on the query DNA, protein, translated RNA, or translated DNA'
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question."
    )
    full_url: str = Field(
        default='TBF',
        description="Url for the BLAT query, use the given examples to make the according one!"
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question."
    )


class BLATdb(BaseModel):
        ucsc_db: str = Field(
        default="hg38",
        description="Genome assembly to use in the UCSC Genome Browser, use the correct db for the organisms. Human:hsg38; Mouse:mm10; Dog:canFam6"
    )
       

class AnswerExtractor:
    """Extract answer for BLATresults """
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        template_api_eutils = """
        You have to answer the question:{question} as clear and short as possible, be factual!\n\
        Example question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT"
        Output to find answer in: "track": "blat", "genome": "hg38", "fields": ["matches", "misMatches", "repMatches", "nCount", "qNumInsert", "qBaseInsert", "tNumInsert", "tBaseInsert", "strand", "qName", "qSize", "qStart", "qEnd", "tName", "tSize", "tStart", "tEnd", "blockCount", "blockSizes", "qStarts", "tStarts"], "blat": [[128, 0, 0, 0, 0, 0, 0, 0, "+", "YourSeq", 128, 0, 128, "chr15", 101991189, 91950804, 91950932, 1, "128", "0", "91950804"], [31, 0, 0, 0, 1, 54, 1, 73, "-", "YourSeq", 128, 33, 118, "chr6", 170805979, 48013377, 48013481, 2, "14,17", "10,78", "48013377,48013464"], [29, 0, 0, 0, 0, 0, 1, 114, "-", "YourSeq", 128, 89, 118, "chr9", 138394717, 125385023, 125385166, 2, "13,16", "10,23", "125385023,125385150"], [26, 1, 0, 0, 0, 0, 1, 2, "+", "YourSeq", 128, 1, 28, "chr17", 83257441, 62760282, 62760311, 2, "5,22", "1,6", "62760282,62760289"], [24, 3, 0, 0, 0, 0, 0, 0, "-", "YourSeq", 128, 54, 81, "chr11_KI270832v1_alt", 210133, 136044, 136071, 1, "27", "47", "136044"], [20, 0, 0, 0, 0, 0, 0, 0, "+", "YourSeq", 128, 106, 126, "chr2", 242193529, 99136832, 99136852, 1, "20", "106", "99136832"]]\
        Answer: chr15:91950805-91950932\n\
        Based on the information given here:\n\
        {context}
        """
        self.eutils_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template_api_eutils)
    def query(self,  question: str, file_path: str) -> str:
        #we make a short extract of the top hits of the files
        first_400_lines = []
        with open(file_path, 'r') as file:
            for _ in range(400):
                line = file.readline()
                if not line:
                    break
                first_400_lines.append(line)
        # Create a temporary file and write the lines to it
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.writelines(first_400_lines)
            temp_file_path = temp_file.name       
        if os.path.exists(temp_file_path):
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
        ncbi_qa_chain= ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=doc_embeddings.as_retriever(), 
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.eutils_CHAIN_PROMPT},
            verbose=True,
        )
        relevant_api_call_info = ncbi_qa_chain(question)
        return relevant_api_call_info
       
def BLAT_api_query_generator(question: str):
    """FUNCTION to write api call for any BLAT query, """
    ucsc_retriever = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/ucsc_genomes", embedding)
    BLAT_structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input, for human always use hg38: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    runnable_BLATQueryRequest = create_structured_output_runnable(BLATQueryRequest, llm, BLAT_structured_output_prompt)
    runnable_BLATdb = create_structured_output_runnable(BLATdb, llm, BLAT_structured_output_prompt)
    #retrieve relevant info to question
    retrieved_docs_data_base = ucsc_retriever.as_retriever().get_relevant_documents(question)
    retrieved_docs_query = ucsc_retriever.as_retriever().get_relevant_documents(question)
    # print(retrieved_docs_data_base[0])
    #keep top 3 hits
    top_3_retrieved_docs = ''.join(doc.page_content for doc in retrieved_docs_query[:3])
    BLAT_db_obj = runnable_BLATdb.invoke({"input": f"User question = {question}\nexample documentation: {retrieved_docs_data_base}"})
    print(f'Database used:{BLAT_db_obj.ucsc_db}')
    BLAT_call_obj = runnable_BLATQueryRequest.invoke({"input": f"User question = {question}\nexample documentation: {top_3_retrieved_docs}"})
    BLAT_call_obj.ucsc_db = BLAT_db_obj.ucsc_db
    BLAT_call_obj.question_uuid=str(uuid.uuid4())
    data = {
        # 'url' : 'https://genome.ucsc.edu/cgi-bin/hgBlat?',
        'userSeq' : BLAT_call_obj.query,
        'type': BLAT_call_obj.ucsc_query_type,
        'db': BLAT_call_obj.ucsc_db,
        'output': 'json'
    }
    # Make the API call
    query_string = urlencode(data)
    # Combine base URL with the query string
    full_url = f"{BLAT_call_obj.url}?{query_string}"
    BLAT_call_obj.full_url = full_url
    BLAT_call_obj.question_uuid = str(uuid.uuid4())
    return BLAT_call_obj


def BLAT_API_call_executer(request_data: BLATQueryRequest):
    """Define
    """
    print('In API caller function\n--------------------')
    print(request_data)
    # Default values for optional fields
    default_headers = {"Content-Type": "application/json"}
    default_method = "GET"
    req = urllib.request.Request(request_data.full_url, headers=default_headers, method=default_method)
    try:
        with urllib.request.urlopen(req) as response:
            response_data = response.read()
            #some db efetch do not return data as json, but we try first to extract the json
            try:
                return json.loads(response_data)
            except:
                return response_data
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        try:
            with urllib.request.urlopen(req) as response:
                response_data = response.read()
                #some db efetch do not return data as json, but we try first to extract the json
                try:
                    if request_data.retmode.lower() == "json":
                        return json.loads(response_data)
                except:
                    return response_data
        except:
            print('error not fixed')
            return f"HTTP Error: {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return f"URL Error: {e.reason}"

def save_BLAT_result(query_request, BLAT_response, file_path):
    """Function saving BLAT results and returns file_name"""
    try:
        # Set file name and construct full file path
        file_name = f'BLAT_results_{query_request.question_uuid}.json'
        full_file_path = os.path.join(file_path, file_name)

        # Open the file for writing
        with open(full_file_path, 'w') as file:
            # Write the static parts of the BLAT_response
            for key in BLAT_response:
                if key != 'blat':
                    json.dump({key: BLAT_response[key]}, file)
                    file.write('\n')

            # Write each list inside the 'blat' key on a new line
            for blat_entry in BLAT_response['blat']:
                json.dump(blat_entry, file)
                file.write('\n')

        return file_name
    # try:
    #     # Set file name and construct full file path
    #     file_name = f'BLAT_results_{query_request.question_uuid}.json'
    #     full_file_path = os.path.join(file_path, file_name)
    #     # Try to save as JSON
    #     with open(full_file_path, 'w') as file:
    #         json.dump(BLAT_response, file, indent=None)
    #         return file_name
    except Exception as e:
        # print(f"Error saving as JSON: {e}")
        # Determine the type of BLAT_response and save accordingly
        if isinstance(BLAT_response, bytes):
            file_name = f'BLAT_results_{query_request.question_uuid}.bin'
        elif isinstance(BLAT_response, str):
            file_name = f'BLAT_results_{query_request.question_uuid}.txt'
        elif isinstance(BLAT_response, dict) or isinstance(BLAT_response, list):
            file_name = f'BLAT_results_{query_request.question_uuid}.json'
        else:
            file_name = f'BLAT_results_{query_request.question_uuid}.json'
        # Update the full file path
        full_file_path = os.path.join(file_path, file_name)
        print(f'\nFull_file_path:{full_file_path}')
        # Save the file
        with open(full_file_path, 'wb' if isinstance(BLAT_response, bytes) else 'w') as file:
            if isinstance(BLAT_response, bytes):
                file.write(BLAT_response)
            elif isinstance(BLAT_response, str) or not isinstance(BLAT_response, dict):
                file.write(BLAT_response if isinstance(BLAT_response, str) else str(BLAT_response))
            else:
                file.write(json.dumps(BLAT_response))
            return file_name

def BLAT_answer(log_file_path, question):
    with open(log_file_path, 'r') as file:
        data = json.load(file)
    current_uuid = data[-1]['uuid']
    print(current_uuid)
    # Access the last entry in the JSON array
    last_entry = data[-1]
    # Extract the file path
    current_file_path = last_entry['file_path']
    print('3: Extracting answer')
    answer_extractor = AnswerExtractor()
    result = answer_extractor.query(question, current_file_path)
    for entry in data:
        if entry['uuid'] == current_uuid:
            entry['answer'] = result['answer']
            break
    with open(log_file_path, 'w') as file:
        json.dump(data, file, indent=4)  
    return result['answer']


# @tool
def BLAT_tool(question: str):
    """BLAT TOOL, use this tool if you need to BLAT a dna sequence on the BLAT data base on ncbi"""
    print('Executing: BLAT Tool')
    log_file_path='/usr/src/app/baio/data/output/BLAT/logfile.json'
    file_path='/usr/src/app/baio/data/output/BLAT/'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    #generate api call
    query_request = BLAT_api_query_generator(question)
    print(query_request)
    BLAT_response = BLAT_API_call_executer(query_request)
    print(BLAT_response)
    file_name = save_BLAT_result(query_request, BLAT_response, file_path)
    log_question_uuid_json(query_request.question_uuid, question, file_name, file_path, log_file_path, query_request.full_url)
    result = BLAT_answer(log_file_path, question)
    return result