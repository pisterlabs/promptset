# Init
import os
import re
import json
import datetime
import ast
import tiktoken
from dotenv import load_dotenv

import openai
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import numpy as np
import pandas as pd

class CodeGenerator():
    def __init__(self, model, temperature, df_dict):
        load_dotenv()
        self.codegen_key = os.getenv('OPENAI_API_KEY_CODEGEN')
        
        # Init
        self.dfs = list(df_dict.values())
        self.retry_counter = 3
        
        # Prompt to generate code
        self.prompt = PromptTemplate.from_template("""
        You're a expert Python coder specializing in Pandas library.                                      
                                              
        Generate function handle_data(dfs) written in Python to process Pandas dataframes 
        following the given guide. The parameter "dfs" is a list of dataframes.
        
        The given guide:
        {guide}
 
        The generated code must be within ```python\ndef handle_data(dfs):\n...```.
        
        Remember to only return the Python code without any other comments.
        """)
        
        llm = ChatOpenAI(model=model,
                         temperature=temperature)

        self.chain = LLMChain(llm=llm,
                              prompt=self.prompt,
                              verbose=False)
        
        # Prompt to retry when code fails to run
        self.retry_prompt = PromptTemplate.from_template("""
        You're a expert Python coder specializing in Pandas library.
        
        Given a piece of code that raised one or more errors and a guide.
        Your task is to fix the given code to fix the errors and ensure that the fixed code can
        still satisfy the guide.    
        
        The given code:
        {code}
        
        The error that the given code raised:
        {error}
        
        The given guide:
        {guide}
 
        The fixed code must be within ```python\ndef handle_data(dfs):\n...```.
        
        Remember to only return the Python code without any other comments.
        """)

        retry_llm = ChatOpenAI(model=model,
                               temperature=temperature)
        
        self.retry_chain = LLMChain(llm=retry_llm,
                                    prompt=self.retry_prompt,
                                    verbose=False)      
    
    def count_token(self, text, model="gpt-4"):
        encoder = tiktoken.encoding_for_model(model)
        tokens = len(encoder.encode(text))
        return tokens
    
    def check_code(self, code):
        try:
            parsed = ast.parse(code)
            for node in ast.walk(parsed):
                if isinstance(node, ast.FunctionDef):
                    return True
        except SyntaxError:
            return False
    
    def handle_output(self, output):
        if isinstance(output, str):
            try:
                output = eval(output)
            except Exception as e:
                print(e)
                try:
                    output = pd.read_json(output, encoding="utf-8")
                except Exception as e:
                    print(e)
            
        if isinstance(output, dict):
            for ele in output:
                if isinstance(output[ele], list) or isinstance(output[ele], tuple):
                    output[ele] = pd.DataFrame(list(output[ele]))
                elif isinstance(output[ele], np.ndarray): 
                    output[ele] = pd.DataFrame(output[ele])
                elif isinstance(output[ele], pd.Series):
                    output[ele] = output[ele].to_frame()
                if isinstance(output[ele], pd.DataFrame):
                    output[ele] = output[ele].fillna(0)
                    output[ele] = output[ele].head(50).to_string()
                else:
                    output[ele] = str(output[ele])
        else:
            if isinstance(output, list) or isinstance(output, tuple):
                output = pd.DataFrame(list(output))
            elif isinstance(output, np.ndarray):
                output = pd.DataFrame(output)
            elif isinstance(output, pd.Series):
                output = output.to_frame()
            
            if isinstance(output, pd.DataFrame):
                output = output.fillna(0)
                output = output.head(200)
            else:
                output = str(output)
        return output
        
    def run(self, guide):
        try:
            code_in = self.prompt.format(guide=guide)
            code = self.chain({"guide": guide})['text']
        except Exception as e:
            print(e)
            print("ERROR CODEGEN - GPT")
            return True, ""
        
        match = re.search(r'def(\s+)(.*?)\((.*?)\):\n((.*?)\n)*(\s+)return(\s+)(.*?)\n', code)
        if match:
            code = match.group()
            if not self.check_code(code):
                print("ERROR CODEGEN - AST")
                return True, ""
        else:
            print("ERROR CODEGEN - REGEX")
            return True, ""
        
        code_out = code
        
        print("########## CODE ##########")
        print(code)
        print("########## CODE ##########")
        
        code = re.sub(r'def (.*?)\(', "def handle_data(", code)    
        code = re.sub(r'to_datetime\(format=\'(.*?)\'', "to_datetime(format='mixed'", code)
        
        current_retry = 0
        while True:            
            try:
                globals_dict = globals()
                locals_dict = {}
                exec(code, globals_dict, locals_dict)
                function = locals_dict['handle_data']
                result = function(self.dfs)
                break
            except Exception as e:
                current_retry += 1
                print(e)
                print(f"ERROR CODERUN {current_retry}")
                
                if current_retry > self.retry_counter:
                    print("ERROR CODERUN - OVER RETRY LIMIT")
                    return True, ""
                
                try:
                    retry_in = self.retry_prompt.format(code=code,
                                                        error=e,
                                                        guide=guide)
                    code = self.retry_chain({"code" : code,
                                             "error" : e,
                                             "guide" : guide})['text']
                    
                    code_in += "\n###\n" + retry_in
                    code_out += "\n###\n" + code
                except Exception as e:
                    print(e)
                    print("ERROR CODEGEN_RETRY - GPT")
                    return True, ""   
                
                match = re.search(r'def(\s+)(.*?)\((.*?)\):\n((.*?)\n)*(\s+)return(\s+)(.*?)\n', code)
                if match:
                    code = match.group()
                    if not self.check_code(code):
                        print("ERROR CODEGEN_RETRY - AST")
                        return True, ""
                else:
                    print("ERROR CODEGEN_RETRY - REGEX")
                    return True, ""
                
                print(f"########## CODE RETRY {current_retry} ##########")
                print(code)
                print(f"########## CODE RETRY {current_retry} ##########")
                
                continue 
            
        try:
            output = self.handle_output(result)
        except Exception as e:
            print(e)
            print("ERROR HANDLE OUTPUT")
            return True, ""
        
        print("########## OUTPUT ##########")
        print(output)
        print("########## OUTPUT ##########")
        
        code_details = {'code_in': code_in,
                        'code_in_token': self.count_token(code_in),
                        'code_out': code_out,
                        'code_out_token': self.count_token(code_out)}
        
        # print(code_details)
        
        return False, output, code_details
     
