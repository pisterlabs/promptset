import argparse
import os
import openai
import ast
import sys
import subprocess
from helper import parse_body_to_return,get_accepted_answer
from pipeline import load_json_file
import logging

from bs4 import BeautifulSoup


#Need the environment to have openai api key beforehand.
openai.api_key = os.getenv("OPEN_AI_API_KEY")


class SimpleClassVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_names = []

    def visit_FunctionDef(self, node):
        self.function_names.append(node.name)
        ast.NodeVisitor.generic_visit(self, node)





def parse_to_get_function_name_list(code_snippet:str):
    """
    Return function names in a given python code snippet.
    """
    visitor = SimpleClassVisitor()
    parsed = ast.parse(code_snippet)
    visitor.visit(parsed)
    return visitor.function_names


def generated_codex_code_test_case(code_snippet:str,**kwargs)-> dict:
    """
    Return the codex code generated from a given python code snippet.
    
    args:
        code_snippet: str
            The python code snippet to be augmented with test cases.

    returns:
        str: The codex test case code generated from the given python code snippet.

    """
    try:
        function_name_list = parse_to_get_function_name_list(code_snippet)
        function_name_prompt = ", ".join(function_name_list)
        prompt_format = f"#Generate test cases for {function_name_prompt} function arguments.\n"
        response = openai.Edit.create(
        engine="code-davinci-edit-001",
        input=code_snippet,
        instruction=prompt_format,
        temperature = 0.7,
        top_p = 1,
        **kwargs
        )
        response_dict = {
            "response" :response,
            "prompt" : prompt_format
        }
        return response_dict
    except:
        return {
            "response" : None,
            "prompt" : None
        }

def generated_codex_code_type_inference(code_snippet:str,**kwargs)-> dict:
    """
    Return the codex code generated from a given python code snippet.
    
    args:
        code_snippet: str
            The python code snippet to be augmented with test cases.

    returns:
        str: The codex test case code generated from the given python code snippet.

    """
    try:
        function_name_list = parse_to_get_function_name_list(code_snippet)
        function_name_prompt = ", ".join(function_name_list)
        prompt_format = f"Anotate mypy Types for {function_name_prompt} function arguments.\n"
        response = openai.Edit.create(
        engine="code-davinci-edit-001",
        input=code_snippet,
        instruction=prompt_format,
        temperature = 0.7,
        top_p = 1,
        **kwargs
        )
        response_dict = {
            "response" :response,
            "prompt" : prompt_format
        }
        return response_dict
    except:
        return {
            "response" : None,
            "prompt" : None
        }


class TypeChecker:
    def __init__(self) -> None:
        self.tmp_dir = "tmp/"
        os.makedirs(self.tmp_dir, exist_ok=True)
    
    def check_runtime(self,code_snippet:str):
        raise NotImplementedError

    def eval_type_file(self,code_snippet:str)->str:
        """
        Given a code snippet type checks and hashes the stdout
        """
        file_dir = os.path.join(self.tmp_dir,"tmp.py")
        with open(file_dir, "w") as f:
            f.write(code_snippet)
        command = [f"mypy","--ignore-missing-imports",file_dir]
        result = subprocess.run(command,stdout=subprocess.PIPE).stdout.decode('utf-8')#(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        os.remove(file_dir)
        return result



class ProcessDataset:
    """
    Process Dataset for Async Augmentations
    """
    def __init__(self,dataset_path:str="dataset/CodeReviewSE.json") -> None:
        self.dataset_path = dataset_path
        self.data = load_json_file(self.dataset_path)
        self.type_checker = TypeChecker()
    def __call__(self,index:int=10):
        """
        Given a index, return the corresponding data entry.
        """
        data_point =  self.data[str(index)]
        code_blocks_data_point = parse_body_to_return(data_point["body"])
        return code_blocks_data_point

        


if __name__ == "__main__":
    dataset = ProcessDataset()
    dataset(1)