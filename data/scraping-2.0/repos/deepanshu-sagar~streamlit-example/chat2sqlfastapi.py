from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from typing import Any, Dict
from sql_connector import SqlConnector
from langchain.llms import OpenAI
from collections import namedtuple
import os
import subprocess
import tempfile
import time

class Input(BaseModel):
    prompt: str
    api_key: str
    featureName: str

class PromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)

def run_cmd(prompt_text):
    # Create a temporary file and write the prompt_text into it
    prompt_fd, prompt_path = tempfile.mkstemp()
    try:
        with os.fdopen(prompt_fd, 'w') as prompt_file:
            prompt_file.write(prompt_text)
        
        # Execute the external command with the path to the temporary file as a parameter
        cmd = ["cat", prompt_path, "|", "bito"]
        print (cmd)
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess returned error: {e.output}")
        output = e.output
    finally:
        # Ensure the temporary file is deleted even if an error occurs
        os.unlink(prompt_path)
    
    return output

class ChatSql:

    def __init__(self, featureName: str, api_key: str) -> None:
        #self.conf: Dict[str, Any] = read_json([f"conf.json"])
        self.llm: object = OpenAI(openai_api_key=api_key)  
        self.info: str = str(read_json(get_final_path(1, [f"{featureName}_info.json"])))
        self.data: str = str(read_txt(get_final_path(1, [f"{featureName}_data.txt"])))

    def prompt_to_query(self, prompt: str) -> Dict[str, str]:
        info = self.info
        data= self.data
        template = """
        Your mission is convert SQL query from given {prompt}. Use following database information for this purpose (info key is a database column name and info value is explanation) : {info} .  along with this i am sharing some sample data from this table :  {data}
        --------
        Put your query in the  JSON structure with key name is 'query'
        """
        pr_ = PromptTemplate(input_variables=["prompt", "info"], template=template)
        final_prompt = pr_.format(
            prompt=prompt,
            info=info,
            data=data
        )
        gpt_query: Dict["str", "str"] = json.loads(self.llm(final_prompt))

        return gpt_query
    
    def prompt_to_query_new(self, prompt: str):
        info = self.info
        data= self.data
        template = """
        Your mission is convert SQL query from given {prompt}. Use following database information for this purpose (info key is a database column name and info value is explanation) : {info} .  along with this i am sharing some sample data from this table :  {data}
        --------
        Put your query in the  JSON structure with key name is 'query'
        """
        pr_ = PromptTemplate(input_variables=["prompt", "info"], template=template)
        final_prompt = pr_.format(
            prompt=prompt,
            info=info,
            data=data
        )
        output = run_cmd(final_prompt)
        return output

    # def run_cmd(prompt_text):
    #     start_time = time.time()
        
    #     # Create a temporary file and write the prompt_text into it
    #     prompt_fd, prompt_path = tempfile.mkstemp()
    #     try:
    #         with os.fdopen(prompt_fd, 'w') as prompt_file:
    #             prompt_file.write(prompt_text)
            
    #         # Execute the external command with the path to the temporary file as a parameter
    #         cmd = ["bito", "-p", prompt_path]
    #         output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Subprocess returned error: {e.output}")
    #         output = e.output
    #     finally:
    #         # Ensure the temporary file is deleted even if an error occurs
    #         os.unlink(prompt_path)
        
    #     end_time = time.time()
    #     total_time = end_time - start_time
    #     return output
            
        


    def query_to_result(self, gpt_query: Dict[str, str]) -> str:
        raw_res: str = SqlConnector(query=gpt_query["query"]).query
        return raw_res

    def raw_result_to_processed(self, raw_result: str) -> str:
        res_processing_template = """
        Your mission is convert database result to meaningful sentences. Here is the database result: {database_result}. validate the query before responding.
        """
        db_pr = PromptTemplate(
            input_variables=["database_result"], template=res_processing_template
        )
        final_prompt = db_pr.format(database_result=raw_result)
        procesed_result: str = self.llm(final_prompt)

        return procesed_result

def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def read_txt(file_path: str) -> str:
    with open(file_path, "r") as file:
        data = file.read()
    return data

def save_json(file_path: str, data: Dict[str, Any]):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def get_final_path(path_type: int, paths: list) -> str:
    current_directory = os.getcwd()
    final_path = os.path.join(current_directory, *paths)
    return final_path

app = FastAPI()

@app.post("/process")
async def process(input: Input):
    csql = ChatSql(input.featureName,input.api_key)
    csql.llm = OpenAI(openai_api_key=input.api_key)  # update the API Key
    query = csql.prompt_to_query_new(input.prompt)
    print (query)
    #result = {"Generated SQL Query": query["query"]}
    return query
