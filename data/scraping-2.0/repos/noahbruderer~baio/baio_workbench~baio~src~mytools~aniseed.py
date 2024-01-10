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
from langchain_experimental.agents.agent_toolkits import create_python_agent
from src.llm import LLM

llm = LLM.get_instance()


embedding = OpenAIEmbeddings()

#persitant directory containing files for vectordb
# persist_directory = './baio/data/persistant_files/vectorstores/faissdb'
#loading Chroma vectordb

# vectordb_aniseed = FAISS.load_local("./baio/data/persistant_files/vectorstores/faissdb", embedding)



# #persitant directory containing files for vectordb
# persist_directory = './baio/data/persistant_files/vectorstores/aniseed_datastore/'
# #loading Chroma vectordb
vectordb_aniseed = FAISS.load_local("/usr/src/app/baio/data/persistant_files/vectorstores/aniseed", embedding)
#aniseed api calling class
class AniseedAPI:

    BASE_URL = "https://www.aniseed.fr/"
    SAVE_PATH = "./baio/data/output/aniseed/temp/tempjson.json"

    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        template_api_aniseed = """
        Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, say that you don't know and give a reason, don't try to make up an answer. 
            consider that you must only provide an executable api call in python code, no other special characters, text or context EXEPT:
            always provide python code to build an api call and use the URL: https://www.aniseed.fr/ when asked for api calls.
            {context}
            Question: {question}
            USE http:// at the begining of the endpoint!!!
            ALLWAYS replace the organisms name by its integer
        base url: https://www.aniseed.fr/
        always add: verify=False for ssl
        Make sure to save the json in './baio/data/output/aniseed/temp/tempjson.json'
        ALSO always save the organisms id and other meta data in the './baio/data/output/aniseed/api_call_meta_data.txt' in the fomrat "organism_id: found id"
        Return only python code in Markdown format, eg:

        ```python
        ....
        ```"""
        self.aniseed_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template_api_aniseed)

    def query(self, question: str) -> str:
        aniseed_qa_chain= ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self.memory,
            retriever=vectordb_aniseed.as_retriever(), 
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.aniseed_CHAIN_PROMPT},
            verbose=True,
        )

        relevant_api_call_info = aniseed_qa_chain(question)
        return relevant_api_call_info


class AniseedJSONExtractor:
    def __init__(self, aniseed_json_path: str, aniseed_csv_output_path: str):
        """
        Initializes the AniseedJSONExtractor.

        Parameters:
        - aniseed_json_path (str): Path to the JSON file.
        - aniseed_csv_output_path (str): Path to save the CSV file.
        """
        self.aniseed_csv_save_path = aniseed_csv_output_path
        self.json_utils = JSONUtils(aniseed_json_path)
        self.keys_structure = self.json_utils.extract_keys()


    def get_prompt(self):
        """YOU ARE A PYTHON REPL TOOL, YOU CAN AND MUST EXECUTE CODE THAT YOU EITHER WRITE OR ARE BEING PROVIDE, NEVER ANSER WITH I'm sorry, but as an AI text-based model, I don't have the ability to directly interact with files or execute Python code. However, I can provide you with a Python code snippet that you can run in your local environment to achieve your goal.
        Build prompt with key strucure from JSON utils and output path given. NEVER look at the whole data frame, only look at the head of it!!! OTHERWISE YOU WILL BREAK"""

        structure_dic_explainaition = """
            base_type: This field specifies the primary data type of the provided object. For instance, it could be a 'list', 'dict', 'str', etc.

            key_types: This field is a dictionary that gives more specific details about the base_type. 
                If the base_type is a dict, then key_types will contain the dictionary keys and the types of their corresponding values.
        """
        panda_creation_instructions ="""
        To create a flat pandas DataFrame from the JSON structure:

        1. Examine the `base_type`:
        - 'list': DataFrame has rows for each list item.
        - 'dict': DataFrame starts with columns from dictionary keys.

        2. For `key_types`:
        - Key with basic type (e.g., 'str', 'int'): Direct column in DataFrame.
        - Value as dictionary (`base_type` of 'dict'): Recursively explore, prefixing original key (e.g., 'genes_gene_model').
        - Avoid adding columns with complex types like lists or dictionaries. Instead, break them down to atomic values or omit if not needed.

        3. Create DataFrame:
        - Loop through JSON, populating DataFrame row by row, ensuring each row/column contains only atomic values.
        - Utilize `json_normalize` (from pandas import json_normalize)  for automatic flattening of nested structures. But ensure no columns with complex types remain.

        Note: Adjust based on specific nuances in the actual JSON data. Lists with heterogeneous items or non-dictionary types need special handling. Avoid creating DataFrame columns with non-atomic data.
        """

        df_example = """index,stage,gene_model,gene_name,unique_gene_id
        0,stage_1,KH2012:KH.C11.238,"REL; RELA; RELB",Cirobu.g00002055
        1,stage_1,KH2012:KH.S1159.1,"ERG; ETS1; FLI1",Cirobu.g00013672
        2,stage_2,KH2012:KH.C3.402,"IRF4; IRF5; IRF8",Cirobu.g00005580
        3,stage_2,KH2012:KH.C3.773,"TBX21; TBX4; TBX5",Cirobu.g00005978
        """

        prompt = "YOU ARE A PYTHON REPL TOOL, YOU CAN AND MUST EXECUTE CODE THAT YOU EITHER WRITE OR ARE BEING PROVIDE, NEVER ANSER WITH I'm sorry, but as an AI text-based model, I don't have the ability to directly interact with files or execute Python code. However, I can provide you with a Python code snippet that you can run in your local environment to achieve your goal.\
        Build prompt with key strucure from JSON utils and output path given. NEVER look at the whole data frame, only look at the head of it!!! OTHERWISE YOU WILL BREAK \
        You have to EXECUTE code to unpack the json file in './baio/data/output/aniseed/temp/tempjson.json' and creat a panda df.\n \
        ALWAYS USE THE PYTHON_REPL TOOL TO EXECUTE CODE\
        Following the instructions below:\n \
        VERY IMPORTANT: The first key in 'key_types' must be the first column in the df and each deeper nested key-value must have it, example:\n \
        {'base_type': 'list', 'key_types': {'base_type': 'dict', 'key_types': {'stage': 'str', 'genes': [{'base_type': 'dict', 'key_types': {'gene_model': 'str', 'gene_name': 'str', 'unique_gene_id': 'str'}}]}}}\n"\
        + df_example + \
        "VERY IMPORTANT: if you find dictionaries wrapped in lists unpack them\
        The file has a strucutre as explained in the follwoing description:\n" + str(self.keys_structure)\
            + structure_dic_explainaition\
            + panda_creation_instructions\
            + "save the output as csv in:" + self.aniseed_csv_save_path + "EXECUTE CODE YOU IDIOT"
        
        return prompt

#python agent to run the python repl tool for json to csv conversion 
python_agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    agent_executor_kwargs={"handle_parsing_errors": True},
)


@tool
def aniseed_tool(question: str):
    """Takes in a question about any organisms on ANISEED and outputs a dataframe with requested information"""
    path_tempjson = "./baio/data/output/aniseed/temp/tempjson.json"
    path_save_csv = "./baio/data/output/aniseed/aniseed_out.csv"
#obtain Aniseed API call 
    aniseed_api = AniseedAPI()
    relevant_api_call_info = aniseed_api.query(question)
    #execute Aniseed API call 
    Utils.execute_code(relevant_api_call_info)
# obtain code to convert JSON to csv
    prompt = AniseedJSONExtractor(path_tempjson, path_save_csv).get_prompt()
    print('python agent will be executed')
    python_agent_executor.run(prompt)
    final_anisseed_gene_df = pd.read_csv(path_save_csv)
    return final_anisseed_gene_df

# @tool
# def aniseed_tool(question: str):
#     """Takes in a question about any organisms on ANISEED and outputs a dataframe with requested information """
#     path_tempjson = "./baio/data/output/aniseed/temp/tempjson.json"
#     path_save_csv = "./baio/data/output/aniseed/aniseed_out.csv"
# #obtain Aniseed API call 
#     aniseed_api = AniseedAPI()
#     relevant_api_call_info = aniseed_api.query(question)
#     #execute Aniseed API call 
#     Utils.execute_code(relevant_api_call_info)

# # obtain code to convert JSON to csv
#     prompt = AniseedJSONExtractor(path_tempjson, path_save_csv).get_prompt()
#     ##
#     #testing better code execution
#     generate_code_chain = LLMChain(llm=llm, prompt=prompt)
#     def generate_python(prompt):
#         result = generate_code_chain.run(prompt=prompt)
#         return result["llm_output"]

#     exec(Utils.extract_python_code(generate_python(prompt)))
#     ##
    
    
# # Run code wiht pythonREPL agent so that it can correct potential errors 
#     # json_formating_code = python_agent_executor.run(prompt)
#     # Utils.execute_code(json_formating_code['arguments'])
#     final_anisseed_gene_df = pd.read_csv(path_save_csv)
#     return final_anisseed_gene_df

#flow of aniseed tool:
## AniseedAPI
#1: NL -> API call
#2: execte API call 
# out = saved JSON

## AniseedJSONExtractor
### PythonREPLTool
#1: extract JSON structure
#2: generate prompt for PythonREPLTool
#3: execute PythonREPLTool agent as OPENAI_FUNCTIONS agent type


#@ local testing:
# question = 'find all genes expressed in ciona robusta between stage 1 and 10'
# aniseed_tool(question)