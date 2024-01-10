import urllib.request
import urllib.parse
import json
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from typing import Optional
from langchain.prompts import (
    PromptTemplate,
)
from typing import Optional, Dict
from pydantic import ValidationError
from pydantic import BaseModel, Field
from typing import Optional, Union
import threading
import json
import uuid
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
from src.non_llm_tools.utilities import log_question_uuid_json
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from langchain.chat_models import ChatOpenAI
from urllib.parse import urlencode
from typing import Union, List
# Lock for synchronizing file access
from src.llm import LLM
llm = LLM.get_instance()

embedding = LLM.get_embedding()
file_lock = threading.Lock()

embedding = OpenAIEmbeddings()
ncbi_jin_db = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/ncbi_jin_db_faiss_index", embedding)

class EutilsAPIRequest(BaseModel):
    url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        description="URL endpoint for the NCBI Eutils API, always use esearch except for db=snp, then use https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi."
    )
    method: str = Field(
        default="GET",
        description="HTTP method for the request. Typically 'GET' or 'POST'."
    )
    headers: Dict[str, str] = Field(
        default={"Content-Type": "application/json"},
        description="HTTP headers for the request. Default is JSON content type."
    )
    db: str = Field(
        ...,
        description="Database to search. E.g., 'gene' for gene database, 'snp' for SNPs, 'omim' for genetic diseases. ONLY ONE to best answer the question"
    )
    retmax: int = Field(
        ...,
        description="Maximum number of records to return."
    )
    retmode: str = Field(
        default="json",
        description="Return mode, determines the format of the response. Commonly 'json' or 'xml'."
    )
    sort: Optional[str] = Field(
        default="relevance",
        description="Sorting parameter. Defines how results are sorted."
    )
    term: Optional[str] = Field(
        None,
        description="Search term. Used to query the database. if it is for a SNP always remove rs before the number"
    )
    id: Optional[int] = Field(
        None,
        description="ONLY for db=snp!!! Identifier(s) in the search query for specific records when looking for SNPs. Obligated integer without the 'rs' prefix, use user question to fill."
    )
    response_format: str = Field(
        default="json",
        description="Expected format of the response, such as 'json' or 'xml'."
    )
    question_uuid: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question."
    )
    full_search_url: Optional[str] = Field(
    default='TBF',
    description="Search url for the first API call -> obtian id's for call n2"
    )


def eutils_API_query_generator(question: str):
    """FUNCTION to write api call for any BLAST query, """
    ncbi_jin_db = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/ncbi_jin_db_faiss_index", embedding)
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
    runnable = create_structured_output_runnable(EutilsAPIRequest, llm, BLAST_structured_output_prompt)
    #retrieve relevant info to question
    retrieved_docs = ncbi_jin_db.as_retriever().get_relevant_documents(question)
    #keep top 3 hits
    top_3_retrieved_docs = ''.join(doc.page_content for doc in retrieved_docs[:3])
    eutils_call_obj = runnable.invoke({"input": f"User question = {question}\nexample documentation: {top_3_retrieved_docs}"})
    eutils_call_obj.question_uuid=str(uuid.uuid4())
    return eutils_call_obj



class NCBIAPI:
    """another test, if fails use ncbiapi2 """
    def query(self, question: str) -> str:
        retriever = ncbi_jin_db.as_retriever()
        retrieved_docs = retriever.invoke(question)
        relevant_api_call_info = retrieved_docs[0].page_content
        return relevant_api_call_info
    
def format_search_term(term, taxonomy_id=9606):
    """To be replaced with chain that fetches the taxonomy id of the requested organism!"""
    if term is None:
        return f"+AND+txid{taxonomy_id}[Organism]"
    else:
        return term + f"+AND+txid{taxonomy_id}[Organism]"

class EfetchRequest(BaseModel):
    url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        description="URL endpoint for the NCBI Efetch API."
    )
    db: str = Field(
        ...,
        description="Database to fetch from. E.g., 'gene' for gene database."
    )
    id: Union[int, str, List[Union[int, str]]] = Field(
        ...,
        description="Comma-separated list of NCBI record identifiers."
    )
    retmode: str = Field(
        default="xml",
        description="Return mode, determines the format of the response. Commonly 'xml' or 'json'."
    )
    full_search_url: Optional[str] = Field(
    default='TBF',
    description="Search url for the efetch API call"
    )
   

class AnswerExtractor:
    """Extract answer for eutils and blast results """
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        template_api_eutils = """
        You have to answer the question:{question} as clear and short as possible manner, be factual!\n\
        Example question: What is the official gene symbol of LMP10?
        Output to find answer in: [b'1. Psmb10 Official Symbol: Psmb10 and Name: proteasome (prosome, macropain) subunit, beta type 10 [Mus musculus (house mouse)] Other Aliases: Mecl-1, Mecl1 Other Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; prosome Mecl1; proteasome (prosomome, macropain) subunit, beta type 10; proteasome MECl-1; proteasome subunit MECL1; proteasome subunit beta-2i\nChromosome: 8; Location: 8 53.06 cM\nAnnotation: Chromosome 8 NC_000074.7 (106662360..106665024, complement)\nID: 19171\n\n2. PSMB10\nOfficial Symbol: PSMB10 and Name: proteasome 20S subunit beta 10 [Homo sapiens (human)]\nOther Aliases: LMP10, MECL1, PRAAS5, beta2i\nOther Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; proteasome (prosome, macropain) subunit, beta type, 10; proteasome MECl-1; proteasome catalytic subunit 2i; proteasome subunit MECL1; proteasome subunit beta 10; proteasome subunit beta 7i; proteasome subunit beta-2i; proteasome subunit beta2i Chromosome: 16; Location: 16q22.1 Annotation: Chromosome 16 NC_000016.10 (67934506..67936850, complement) MIM: 176847 ID: 5699  3. MECL1 Proteosome subunit MECL1 [Homo sapiens (human)] Other Aliases: LMP10, PSMB10 This record was replaced with GeneID: 5699 ID: 8138  ']\
        Answer: PSMB10\n\
        Example question: Which gene is SNP rs1217074595 associated with?
        Output to find answer in: [b   header :  type : esummary , version : 0.3  , result :  uids :[ 1217074595 ], 1217074595 :  uid : 1217074595 , snp_id :1217074595, allele_origin :  , global_mafs :[  study : GnomAD , freq : A=0.000007/1  ,  study : TOPMED , freq : A=0.000004/1  ,  study : ALFA , freq : A=0./0  ], global_population :  , global_samplesize :  , suspected :  , clinical_significance :  , genes :[  name : LINC01270 , gene_id : 284751  ], acc : NC_000020.11 , chr : 20 , handle : GNOMAD,TOPMED , spdi : NC_000020.11:50298394:G:A , fxn_class : non_coding_transcript_variant , validated : by-frequency,by-alfa,by-cluster , docsum : HGVS=NC_000020.11:g.50298395G>A,NC_000020.10:g.48914932G>A,NR_034124.1:n.351G>A,NM_001025463.1:c.*4G>A|SEQ=[G/A]|LEN=1|GENE=LINC01270:284751 , tax_id :9606, orig_build :155, upd_build :156, createdate : 2017/11/09 09:55 , updatedate : 2022/10/13 17:11 , ss : 4354715686,5091242333 , allele : R , snp_class : snv , chrpos : 20:50298395 , chrpos_prev_assm : 20:48914932 , text :  , snp_id_sort : 1217074595 , clinical_sort : 0 , cited_sort :  , chrpos_sort : 0050298395 , merged_sort : 0    \n ]\n\
        Answer: LINC01270\n\
        Example question: What are genes related to Meesmann corneal dystrophy?\n\
        Output to find answer in: [b   header :  type : esummary , version : 0.3  , result :  uids :[ 618767 , 601687 , 300778 , 148043 , 122100 ], 618767 :  uid : 618767 , oid : #618767 , title : CORNEAL DYSTROPHY, MEESMANN, 2; MECD2 , alttitles :  , locus : 12q13.13  , 601687 :  uid : 601687 , oid : *601687 , title : KERATIN 12, TYPE I; KRT12 , alttitles :  , locus : 17q21.2  , 300778 :  uid : 300778 , oid : %300778 , title : CORNEAL DYSTROPHY, LISCH EPITHELIAL; LECD , alttitles :  , locus : Xp22.3  , 148043 :  uid : 148043 , oid : *148043 , title : KERATIN 3, TYPE II; KRT3 , alttitles :  , locus : 12q13.13  , 122100 :  uid : 122100 , oid : #122100 , title : CORNEAL DYSTROPHY, MEESMANN, 1; MECD1 , alttitles :  , locus : 17q21.2    \n ]\
        Answer: KRT12, KRT3\
        Example question:
        Output to find answer in: [b'\n1. PSMB10\nOfficial Symbol: PSMB10 and Name: proteasome 20S subunit beta 10 [Homo sapiens (human)]\nOther Aliases: LMP10, MECL1, PRAAS5, beta2i\nOther Designations:
        Answer: PSMB10
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
    
def api_query_generator(question: str):
    """ NEW VERSION FIRST IN THE PIPE :
    function executing:
    1: text retrieval from ncbi_doc
    2: structured data from (1) to generate EutilsAPIRequest object
    """
    output = eutils_API_query_generator(question)
    try:
        query_request = output
        query_request.url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        if query_request.db == 'gene':
            query_request.term=format_search_term(query_request.term)
            query_request.retmode = 'json'
        if query_request.db == 'snp':
            query_request.url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        print(f'Queryy is: {query_request}')
        #we set the url here, pipeline requires it to be esearch
        # Now you can use query_request as an instance of BlastQueryRequest
    except ValidationError as e:
        print(f"Validation error: {e}")
        return ['Failed to write API query instructions', output]
        # Handle validation error
    return query_request

def make_api_call(request_data: Union[EutilsAPIRequest, EfetchRequest]):
    """Define
    """
    print('In API caller function\n--------------------')
    print(request_data)
    # Default values for optional fields
    default_headers = {"Content-Type": "application/json"}
    default_method = "GET"
    # # Prepare the query string
    # query_params = request_data.dict(exclude={"url", "method", "headers", "body", "response_format", "parse_keys"})
    # if request_data.db == "omim":
    #     query_params = request_data.dict(exclude={"url", "method", "headers", "body", "response_format", "parse_keys", "retmod"})
    #     if request_data.id != '' and request_data.id is not None:
    #         request_data.url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
    if isinstance(request_data, EfetchRequest):
        if request_data.db == "gene":
            print('FETCHING')
            # print(request_data)
            if isinstance(request_data.id, list):
                    id_s = ','.join(map(str, request_data.id))  # Convert each element to string and join
            else:
                    id_s = str(request_data.id)
            query_params = request_data.dict(include={"db", "retmax","retmode"})
            # query_params = {
            # 'db': request_data.db,
            # 'retmax': request_data.retmax,
            # 'retmode': request_data.retmode,
            # # Add other parameters here if needed
            # }
            # print(query_params)            
            # print(id_s)
            encoded_query_string = urlencode(query_params)
            query_string = f"{encoded_query_string}&id={id_s}"
            # encoded_query_string = urlencode(query_params)
            request_data.full_search_url = f"{request_data.url}?{query_string}"
    print(f'Requesting: {request_data.full_search_url}')

    req = urllib.request.Request(request_data.full_search_url, headers=default_headers, method=default_method)
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

def result_file_extractor(question, file_path):
    """Extracting the answer result file"""
    print('In result file extractor')
    #extract answer
    answer_extractor = AnswerExtractor()
    return answer_extractor.query(question, file_path)

@tool
def eutils_tool(question: str):
    """Tool to make any eutils query, creates query, executes it, saves result in file and reads answer"""
    print('Running: Eutils tool')
    max_ids = 5
    file_name = None  # Initialize file_name variable
    file_path = './baio/data/output/eutils/results/files/'
    log_file_path = './baio/data/output/eutils/results/log_file/eutils_log.json'
    # Check if the directories for file_path and log_file_path exist, if not, create them
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    print('1: Building API call for user question:')
    #FIRST API CALL TO GET IDs
    api_call_nr_1 = api_query_generator(question)
    efetch_response_list = []
    if api_call_nr_1.db != 'snp':
        #uuid_to keep track of question & results 
        api_call_nr_1.question_uuid = str(uuid.uuid4())
        #log the question
        response_api_call_nr_1 = make_api_call(api_call_nr_1)
        print(f'2: Executed API call: {api_call_nr_1.full_search_url=}\n----')
        if api_call_nr_1.retmode == 'json':
            id_list = response_api_call_nr_1.get('esearchresult', {}).get('idlist', [])
        if api_call_nr_1.retmode == 'xml':
            id_list = response_api_call_nr_1.get('eSearchResult', {}).get('IdList', []).get('Id', [])
        efetch_request = EfetchRequest(db=api_call_nr_1.db,
                                            id=id_list,
                                            retmode=api_call_nr_1.retmode)
        efetch_response = make_api_call(efetch_request)
        efetch_response_list.append(efetch_response)            
        try:
            # Set file name and construct full file path
            file_name = f'eutils_results_{api_call_nr_1.question_uuid}.json'
            full_file_path = os.path.join(file_path, file_name)
            # Try to save as JSON
            with open(full_file_path, 'w') as file:
                json.dump(efetch_response_list, file, indent=4)
        except Exception as e:
            # print(f"Error saving as JSON: {e}")
            # Determine the type of efetch_response_list and save accordingly
            if isinstance(efetch_response_list, bytes):
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.bin'
            elif isinstance(efetch_response_list, str):
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.txt'
            elif isinstance(efetch_response_list, dict) or isinstance(efetch_response_list, list):
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.json'
            else:
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.json'
            # Update the full file path
            full_file_path = os.path.join(file_path, file_name)
            print(f'\nFull_file_path:{full_file_path}')
            # Save the file
            with open(full_file_path, 'wb' if isinstance(efetch_response_list, bytes) else 'w') as file:
                if isinstance(efetch_response_list, bytes):
                    file.write(efetch_response_list)
                elif isinstance(efetch_response_list, str) or not isinstance(efetch_response_list, dict):
                    file.write(efetch_response_list if isinstance(efetch_response_list, str) else str(efetch_response_list))
                else:
                    file.write(json.dumps(efetch_response_list))
    else:
        api_call_nr_1.question_uuid = str(uuid.uuid4())
        efetch_response = make_api_call(api_call_nr_1)
        efetch_response_list.append(efetch_response)
        try:
            # Set file name and construct full file path
            file_name = f'eutils_results_{api_call_nr_1.question_uuid}.json'
            full_file_path = os.path.join(file_path, file_name)
            # Try to save as JSON
            with open(full_file_path, 'w') as file:
                json.dump(efetch_response_list, file, indent=4)
        except Exception as e:
            print(f"Error saving as JSON: {e}")
            # Determine the type of efetch_response_list and save accordingly
            if isinstance(efetch_response_list, bytes):
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.bin'
            elif isinstance(efetch_response_list, str):
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.txt'
            elif isinstance(efetch_response_list, dict) or isinstance(efetch_response_list, list):
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.json'
            else:
                file_name = f'eutils_results_{api_call_nr_1.question_uuid}.json'
            # Update the full file path
            full_file_path = os.path.join(file_path, file_name)
            print(f'Results are saved in:{full_file_path}')
            # Save the file
            with open(full_file_path, 'wb' if isinstance(efetch_response_list, bytes) else 'w') as file:
                if isinstance(efetch_response_list, bytes):
                    file.write(efetch_response_list)
                elif isinstance(efetch_response_list, str) or not isinstance(efetch_response_list, dict):
                    file.write(efetch_response_list if isinstance(efetch_response_list, str) else str(efetch_response_list))
                else:  # dict or list
                    file.write(json.dumps(efetch_response_list))                     
                      
    log_question_uuid_json(api_call_nr_1.question_uuid, question, file_name, file_path, log_file_path, api_call_nr_1.full_search_url)
    ###extract answer
    with file_lock:
        with open(log_file_path, 'r') as file:
            data = json.load(file)
        current_uuid = data[-1]['uuid']

    # Access the last entry in the JSON array
    last_entry = data[-1]
    # Extract the file path
    current_file_path = last_entry['file_path']
    print('3: Extracting answer')
    result = result_file_extractor(question, current_file_path)
    for entry in data:
        if entry['uuid'] == current_uuid:
            entry['answer'] = result['answer']
            break
    # Write the updated data back to the log file
    with file_lock:
        with open(log_file_path, 'w') as file:
            json.dump(data, file, indent=4)  
    # Call the logging function with the full file path
    print(result['answer'])
    print('EUTILS Tool done')
    return result['answer']

# result_file_extractor('Convert ENSG00000205403 to official gene symbol.', '/usr/src/app/baio/data/output/eutils/results/files/eutils_results_e58ad48e-5ca7-4e98-bd29-d8f4b9eb44c5.json')
# result = eutils_tool('Convert ENSG00000205403 to official gene symbol.')