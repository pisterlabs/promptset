from typing import Optional
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
  ConversationalRetrievalChain
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from src.non_llm_tools.utilities import Utils, JSONUtils
from langchain.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents.agent_toolkits import create_python_agent
import os
from langchain.chat_models import ChatOpenAI
from baio.src.non_llm_tools.utilities import Utils, JSONUtils
from langchain.callbacks import get_openai_callback
from typing import Optional, Sequence

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
  PromptTemplate,
)
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import faiss


class AnswerExtractor:
  def __init__(self):
    self.memory = ConversationBufferMemory(memory_key= chat_history , return_messages=True)

    template_api_ncbi =    
    You have to answer the question: question  in a clear and as short as possible manner, be factual!\n\n
    Based on the information given here:\n
     context 
       
    self.ncbi_CHAIN_PROMPT = PromptTemplate(input_variables=[ context ,  question ], template=template_api_ncbi)
  def query(self, question: str) -> str:
    ncbi_qa_chain= ConversationalRetrievalChain.from_llm(
      llm=llm,
      memory=self.memory,
      retriever=BLAST_results_NTC1DYCC016.as_retriever(), 
      return_source_documents=False,
      combine_docs_chain_kwargs=  prompt : self.ncbi_CHAIN_PROMPT ,
      verbose=True,
    )

    relevant_api_call_info = ncbi_qa_chain(question)
    return relevant_api_call_info
  
#######
#######example code for one file
#######
loader = TextLoader( /usr/src/app/BLAST_results_NTC1DYCC016.txt )
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
BLAST_results_NTC1DYCC016 = FAISS.from_documents(docs, embeddings)

answer_extractor = AnswerExtractor()
answer_rid = answer_extractor.query(question)
#######
#######
#######

#######
#######   Function to process a single file
#######

def process_file(file_path, question):
  #laod
  loader = TextLoader(file_path)
  documents = loader.load()
  #split
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
  docs = text_splitter.split_documents(documents)
  #embed
  doc_embeddings = FAISS.from_documents(docs, embeddings)
  #extract answer
  answer_extractor = AnswerExtractor()
  return answer_extractor.query(question, doc_embeddings)


#######
#######   Loop through the files to 
#######

#store answers as rid:answer
answers =   

#process all files in turing test
files_to_process = [f for f in os.listdir( /usr/src/app/baio/data/persistant_files/evaluation/blast_question_answers/ ) if f.startswith( BLAST_results_ ) and f.endswith( .txt )]

# Reverse the dictionary to get a rid: question mapping
rid_to_question =  rid: question for question, rid in answer_dic.items() 

#store answers
answers =   
directory =  /usr/src/app/baio/data/persistant_files/evaluation/blast_question_answers/ 
files_to_process = [f for f in os.listdir(directory) if f.startswith( BLAST_results_ ) and f.endswith( .txt )]

for file_name in files_to_process:
  file_path = os.path.join(directory, file_name)
  rid = file_name.split( _ )[2].split( . )[0] # Extract RID from file name
  
  # Retrieve the corresponding question for the RID
  if rid in rid_to_question:
    question = rid_to_question[rid]
    if os.path.exists(file_path):
      try:
        answer_rid = blast_result_file_extractor(file_path, question)
        answers[question] = answer_rid
      except Exception as e:
        print(f Error processing file  file_name :  e  )
        answers[question] = Failed 

# Save results to JSON
json_path = os.path.join(directory, gene_turing_test_blast_answers.json )
with open(json_path, w ) as json_file:
  json.dump(answers, json_file)

print( Finished processing files and saved answers. )




###Evaluation FOR BLAST:
import json 
with open( /usr/src/app/baio/data/persistant_files/evaluation/geneturing.json , r ) as file:
  data = json.load(file)

    
# Now answer_dic will contain your questions as keys and their corresponding RIDs as values
answer_dic =   
rid_list = []
error_list = []

for category, questions in data.items():
  if category in [ Multi-species DNA aligment , Human genome DNA aligment ]:
    print(category)
    counter = 0
    for question, answer in questions.items():
      counter +=1
      print(f Question: counter  in  category  )
      query_request = api_query_generator(question)
      print(query_request)
      rid = submit_blast_query(query_request)
      if rid:
        answer_dic[question] = rid # Add question and RID to the dictionary
        rid_list.append(rid)
        try:
          fetch_and_save_blast_results(rid)
        except Exception as e:
          print(f Error:  rid : e  )
          error_list.append(rid)


with open( /usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/ucsc/genomes.json , r ) as file:
  data = json.load(file)
ucsc_genomes_dict = data[ ucscGenomes ]

new_dict =   
for key, value in ucsc_genomes_dict.items():
  new_key = value[ scientificName ]
  new_value = value
  new_value[ originalKey ] = key # Adding the original key to the value
  new_dict[new_key] = new_value
  
  
import csv
flattened_data = []
for key, value in ucsc_genomes_dict.items():
  # Flatten each entry into a single dictionary
  flat_entry =   ucscGenomes : key 
  flat_entry.update(value)
  flattened_data.append(flat_entry)

# Define the CSV file name
csv_file =  /usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/ucsc/ucsc_genomes_converted.csv 

# Write to CSV
with open(csv_file, mode= w , newline= , encoding= utf-8 ) as file:
  # Create a CSV writer object
  writer = csv.DictWriter(file, fieldnames=flattened_data[0].keys())
  
  # Write the header
  writer.writeheader()

  # Write the rows
  for row in flattened_data:
    writer.writerow(row)

print( CSV file created: , csv_file)


import csv

# File paths
input_file =  /usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/ucsc/ucsc_genomes_converted.csv 
output_file =  /usr/src/app/baio/data/persistant_files/user_manuals/api_documentation/ucsc/ucsc_genomes_reduced.csv 

# Columns to keep
columns_to_keep = [ organism , scientificName , originalKey ,  genome , taxId ]

# Read the CSV file and extract the necessary columns
with open(input_file, r ) as infile, open(output_file, w , newline= ) as outfile:
  reader = csv.DictReader(infile)
  writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
  
  writer.writeheader()
  
  for row in reader:
    reduced_row =  key: row[key] for key in columns_to_keep 
    writer.writerow(reduced_row)
    
    
ucsc_genomes = data[ ucscGenomes ]  
with open(output_file, w , newline= ) as file:
  writer = csv.writer(file)
  # Write the header
  header = [ organism , scientificName , originalKey ,  genome , taxId ]
  writer.writerow(header)
  
  # Write each entry
  for key, value in ucsc_genomes.items():
    row = [value.get(field,  ) for field in header]
    row.append(key) # Add the original key as the last column
    writer.writerow(row)
    template_api_ncbi = """You have to answer the question: question  as clear and short as possible manner, be factual!\n\
        USE THE CONTEXT AND THE BELOW ANSWERS TO ANSWER:  context 
        Output to find answer in: [b   header :  type : esummary , context  version : 0.3  , result :  uids :[ 1217074595 ], 1217074595 :  uid : 1217074595 , snp_id :1217074595, allele_origin :  , global_mafs :[  study : GnomAD , freq : A=0.000007/1  ,  study : TOPMED , freq : A=0.000004/1  ,  study : ALFA , freq : A=0./0  ], global_population :  , global_samplesize :  , suspected :  , clinical_significance :  , genes :[  name : LINC01270 , gene_id : 284751  ], acc : NC_000020.11 , chr : 20 , handle : GNOMAD,TOPMED , spdi : NC_000020.11:50298394:G:A , fxn_class : non_coding_transcript_variant , validated : by-frequency,by-alfa,by-cluster , docsum : HGVS=NC_000020.11:g.50298395G>A,NC_000020.10:g.48914932G>A,NR_034124.1:n.351G>A,NM_001025463.1:c.*4G>A|SEQ=[G/A]|LEN=1|GENE=LINC01270:284751 , tax_id :9606, orig_build :155, upd_build :156, createdate : 2017/11/09 09:55 , updatedate : 2022/10/13 17:11 , ss : 4354715686,5091242333 , allele : R , snp_class : snv , chrpos : 20:50298395 , chrpos_prev_assm : 20:48914932 , text :  , snp_id_sort : 1217074595 , clinical_sort : 0 , cited_sort :  , chrpos_sort : 0050298395 , merged_sort : 0    \n ]\n\
        Example question: What is the official gene symbol of LMP10?
        Output to find answer in: [b \n1. Psmb10\nOfficial Symbol: Psmb10 and Name: proteasome (prosome, macropain) subunit, beta type 10 [Mus musculus (house mouse)]\nOther Aliases: Mecl-1, Mecl1\nOther Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; prosome Mecl1; proteasome (prosomome, macropain) subunit, beta type 10; proteasome MECl-1; proteasome subunit MECL1; proteasome subunit beta-2i\nChromosome: 8; Location: 8 53.06 cM\nAnnotation: Chromosome 8 NC_000074.7 (106662360..106665024, complement)\nID: 19171\n\n2. PSMB10\nOfficial Symbol: PSMB10 and Name: proteasome 20S subunit beta 10 [Homo sapiens (human)]\nOther Aliases: LMP10, MECL1, PRAAS5, beta2i\nOther Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; proteasome (prosome, macropain) subunit, beta type, 10; proteasome MECl-1; proteasome catalytic subunit 2i; proteasome subunit MECL1; proteasome subunit beta 10; proteasome subunit beta 7i; proteasome subunit beta-2i; proteasome subunit beta2i\nChromosome: 16; Location: 16q22.1\nAnnotation: Chromosome 16 NC_000016.10 (67934506..67936850, complement)\nMIM: 176847\nID: 5699\n\n3. MECL1\nProteosome subunit MECL1 [Homo sapiens (human)]\nOther Aliases: LMP10, PSMB10\nThis record was replaced with GeneID: 5699\nID: 8138\n\n ]\
        Answer: PSMB10\n\
        Example question: Which gene is SNP rs1217074595 associated with?\n\
        Output to find answer in: [b   header :  type : esummary , version : 0.3  , result :  uids :[ 1217074595 ], 1217074595 :  uid : 1217074595 , snp_id :1217074595, allele_origin :  , global_mafs :[  study : GnomAD , freq : A=0.000007/1  ,  study : TOPMED , freq : A=0.000004/1  ,  study : ALFA , freq : A=0./0  ], global_population :  , global_samplesize :  , suspected :  , clinical_significance :  , genes :[  name : LINC01270 , gene_id : 284751  ], acc : NC_000020.11 , chr : 20 , handle : GNOMAD,TOPMED , spdi : NC_000020.11:50298394:G:A , fxn_class : non_coding_transcript_variant , validated : by-frequency,by-alfa,by-cluster , docsum : HGVS=NC_000020.11:g.50298395G>A,NC_000020.10:g.48914932G>A,NR_034124.1:n.351G>A,NM_001025463.1:c.*4G>A|SEQ=[G/A]|LEN=1|GENE=LINC01270:284751 , tax_id :9606, orig_build :155, upd_build :156, createdate : 2017/11/09 09:55 , updatedate : 2022/10/13 17:11 , ss : 4354715686,5091242333 , allele : R , snp_class : snv , chrpos : 20:50298395 , chrpos_prev_assm : 20:48914932 , text :  , snp_id_sort : 1217074595 , clinical_sort : 0 , cited_sort :  , chrpos_sort : 0050298395 , merged_sort : 0    \n ]\n\
        Answer: LINC01270\n\
        Example question: What are genes related to Meesmann corneal dystrophy?\n\
        Output to find answer in: [b   header :  type : esummary , version : 0.3  , result :  uids :[ 618767 , 601687 , 300778 , 148043 , 122100 ], 618767 :  uid : 618767 , oid : #618767 , title : CORNEAL DYSTROPHY, MEESMANN, 2; MECD2 , alttitles :  , locus : 12q13.13  , 601687 :  uid : 601687 , oid : *601687 , title : KERATIN 12, TYPE I; KRT12 , alttitles :  , locus : 17q21.2  , 300778 :  uid : 300778 , oid : %300778 , title : CORNEAL DYSTROPHY, LISCH EPITHELIAL; LECD , alttitles :  , locus : Xp22.3  , 148043 :  uid : 148043 , oid : *148043 , title : KERATIN 3, TYPE II; KRT3 , alttitles :  , locus : 12q13.13  , 122100 :  uid : 122100 , oid : #122100 , title : CORNEAL DYSTROPHY, MEESMANN, 1; MECD1 , alttitles :  , locus : 17q21.2    \n ]\
        Answer: KRT12, KRT3\
        For any kind of BLAST results use try to use the hit with the best idenity score to answer the questin, if it is not possible move to the next one. \n\
        If you are asked for gene alignments, use the nomencalture as followd: ChrN:start-STOP with N being the number of the chromosome.\n\
        The numbers before and after the hyphen indicate the start and end positions, respectively, in base pairs. This range is inclusive, meaning it includes both the start and end positions.\n\
              Based on the information given here:\n\
        Answer: KRT12, KRT3"""
        
   
   
   
import json

# Load the first JSON file
with open('/usr/src/app/baio/data/output/geneturing_eutils_281123/logfile_snp.json', 'r') as file:
    data1 = json.load(file)

# Load the second JSON file
with open('/usr/src/app/baio/data/persistant_files/evaluation/geneturing.json', 'r') as file:
    data2 = json.load(file)

# Process each item in the first file
for item in data1:
    question = item['question']
    # Search for the question in data2
    for key, value in data2.items():
        if question in value:
            # Add the answer from data2 to the first file's item as 'type answer'
            item['type answer'] = value[question]
            break

# Write the modified data back to the first file or to a new file
with open('/usr/src/app/snp_log_with_type_answers.json', 'w') as file:
    json.dump(data1, file, indent=4)
import json
import csv

# Load the modified JSON data
with open('/usr/src/app/snp_log_with_type_answers.json', 'r') as file:
    data = json.load(file)

# Define the CSV file name
csv_file = '/usr/src/app/snp_ans.csv'

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as file:
    # Create a CSV writer object
    csv_writer = csv.writer(file)
    # Write the header row
    header = data[0].keys()
    csv_writer.writerow(header)
    # Write the JSON data to the CSV file
    for item in data:
        csv_writer.writerow(item.values())

print(f"Data saved to {csv_file}")
