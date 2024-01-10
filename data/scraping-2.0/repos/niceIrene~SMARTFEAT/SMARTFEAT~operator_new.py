import os
import sys
sys.path.append('../')
sys.path.append("./")
sys.path.append('../prompts/')
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from SMARTFEAT.gpt import *
from prompt import *
import json
import re
from SMARTFEAT.serialize import *
import pandas as pd

class Operator(object):
    def __init__(self, agenda,model_prompt, y_attr):
        self.data_agenda = agenda
        self.y_attr = y_attr
        self.model_prompt = model_prompt

class UnaryOperator(Operator):
    # template of unary operator 
    def __init__(self, agenda, model_prompt, y_attr, org_attr):
        super().__init__(agenda, model_prompt, y_attr)
        self.org_attr = org_attr
    def __str__(self):
        return f"Unary operator, original feature '{self.org_attr}' to predict {self.y_attr}."
    def generate_new_feature(self, temp):
        rel_agenda = obtain_rel_agenda([self.org_attr], self.data_agenda)
        y_agenda = obtain_rel_agenda([self.y_attr], self.data_agenda)
        data_prompt = rel_agenda + ', ' + y_agenda + '\n'
        op_prompt = unary_prompt_propose.format(y_attr= self.y_attr, input = self.org_attr)
        prompt = "Dataset feature description: " + data_prompt + "Downstream machine learning models: " + self.model_prompt +'\n'+ op_prompt
        # for propose, generate one candidate. in the answer, it gives a list of proposals.
        res = gpt_propose(prompt = prompt, n = 1, temperature=temp)
        res_lst = []
        for r in res:
            r_lst = r.split()
            if any(substr in r for substr in ['(certain)', '(high)', '(Certain)', '(High)']):
                result_dict = {}
                result_dict['new_feature'] = "{}_{}".format(r_lst[1], self.org_attr)
                result_dict['description'] = r
                result_dict['relevant'] = self.org_attr
                res_lst.append(result_dict)
        if len(res_lst)==0:
            res_lst = None
        print("The set of unary operators to be applied is")
        print(res_lst)
        return res_lst
    def find_function(self, result_dict, temp=0.1):
        # handle one-hot-encoding as specific case:
        if 'encoding' in result_dict['description'] or 'Encoding' in result_dict['description']:
            func_str = 'encoding'
            return func_str
        new_col, rel_cols, descr, rel_agenda = self.parse_output(result_dict)
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["new_feature", "relevant", "description", "rel_agenda"],
        template="You are a data scientist specializing in feature engineering. Your task is to create a transformation function based on the provided input, output, new feature description, while considering input data types and ranges.\
        Generate the most appropriate python function to obtain new feature(s) {new_feature} (output) using feature {relevant} (input), new feature description: {description}, input description: {rel_agenda}. Define the function using the 'def' keyword.")
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"new_feature": new_col, "relevant": rel_cols, "description": descr, "rel_agenda": rel_agenda})
        print(func_str)
        start = func_str.find('def')
        func_str = func_str[start:]
        return func_str
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = obtain_relevant_cols(result_dict)
        rel_agenda = obtain_rel_agenda(rel_cols,self.data_agenda)
        return new_col, descr, rel_cols, rel_agenda        

# used only in Aggregator    
class Descrtizer(UnaryOperator):
    def __str__(self):
        return f"Bucketize '{self.org_attr}' to predict '{self.y_attr}"
    def find_function(self, result_dict, temp=0.1):
        new_col, rel_cols, descr, rel_agenda = self.parse_output(result_dict)
        data_prompt = rel_agenda
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["data_prompt", "new_feature", "relevant", "description"],
        template="You are a data scientist specializing in feature engineering, where you excel in finding the most suitable operation to obtain new features based on attribute context.\
            Attribute description: {data_prompt}, generate the most appropriate python function \
            to obtain new feature {new_feature} (output) using feature {relevant} (input), function description: {description}. Do not provide a lambda function."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"data_prompt": data_prompt, "new_feature": new_col, "relevant": rel_cols, "description": descr})
        start = func_str.find('def')
        func_str = func_str[start:]
        return func_str
    

# binary operator using propose strategy
class BinaryOperator(Operator):
    def __init__(self, agenda, model_prompt, y_attr, org_attrs):
        super().__init__(agenda, model_prompt, y_attr)
        self.org_attrs = org_attrs
    def __str__(self):
        return f"binary operator, using {str(self.org_attrs)}."
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = obtain_relevant_cols(result_dict)
        rel_agenda = obtain_rel_agenda(rel_cols,self.data_agenda)
        return new_col, descr, rel_cols, rel_agenda
    def generate_new_feature(self, temp):
        rel_agenda = obtain_rel_agenda(self.org_attrs, self.data_agenda)
        y_agenda = obtain_rel_agenda([self.y_attr], self.data_agenda)
        data_prompt = rel_agenda + ',' + y_agenda + '\n'
        op_prompt = binary_prompt_propose.format(y_attr= self.y_attr, input = self.org_attrs)
        prompt = "Attribute description: " + data_prompt + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        res = gpt_fix_or_propose_binary(prompt = prompt, n = 1, temperature=temp)
        res_lst = []
        for r in res:
            if 'certain' in r:
                result_dict = {}
                result_dict['new_feature'] = "Binary_{}".format(str(self.org_attrs))
                result_dict['description'] = "Binary operator {}".format(res[0])
                relevant_str = str(self.org_attrs[0]) +',' + str(self.org_attrs[1])
                result_dict['relevant'] = relevant_str
                res_lst.append(result_dict)
        if len(res_lst) == 0:
            res_lst = None
        return res_lst
    def find_function(self, result_dict, temp=0.1):
        new_col, rel_cols, descr, rel_agenda = self.parse_output(result_dict)
        data_prompt = rel_agenda
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["data_prompt", "new_feature", "relevant", "description"],
        template="You are a data scientist specializing in feature engineering, where you excel in finding the most suitable operation to obtain new features based on attribute context.\
            Attribute description: {data_prompt}, generate the most appropriate python function with +/-/*//to obtain new feature {new_feature} (output) using features {relevant} (input), function description: {description}. If the selected attribute is /, Handle the case of devide by zero."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"data_prompt": data_prompt, "new_feature": new_col, "relevant": rel_cols, "description": descr})
        start = func_str.find('def')
        func_str = func_str[start:]
        return func_str

# binary operator using sampling strategy
class BinaryOperatorAlter(Operator):
    def __init__(self, agenda, model_prompt, y_attr, num_samples):
        super().__init__(agenda, model_prompt, y_attr)
        self.num_samples = num_samples
    def __str__(self):
        return f"binary operator."
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = result_dict['relevant']
        rel_agenda = obtain_rel_agenda(rel_cols,self.data_agenda)
        return new_col, descr, rel_cols, rel_agenda
    def generate_new_feature(self, temp):
        op_prompt = binary_prompt_sampling.format(y_attr= self.y_attr)
        prompt = "Dataset feature description: " + self.data_agenda + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt        
        answer = gpt_sampling(prompt = prompt, n = self.num_samples, temperature=temp)
        res_lst = []
        for i in range(len(answer)):
            res_dic = re.search(r'\{[^}]+\}', answer[i])
            answer_dict = eval(res_dic.group(0))
            res_lst.append(answer_dict)
        print("Result list is")
        print(res_lst)
        return res_lst
    def find_function(self, result_dict, temp=0.1):
        new_col, descr, rel_cols, rel_agenda = self.parse_output(result_dict)
        llm = OpenAI(temperature=temp)
        prompt = PromptTemplate(
        input_variables=["new_feature", "relevant", "description", "rel_agenda"],
        template="You are a data scientist specializing in feature engineering. Your task is to create a transformation function based on the provided input, output, new feature description, while considering input data types and ranges.\
            Generate the most appropriate python function with +/-/*//to obtain new feature {new_feature} (output) using features {relevant} (input), new feature description: {description},  input description: {rel_agenda}. \
             Define the function using the 'def' keyword. If the selected attribute is /, Handle the case of devide by zero."
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        func_str = chain.run({"new_feature": new_col, "relevant": rel_cols, "description": descr, "rel_agenda": rel_agenda})
        start = func_str.find('def')
        func_str = func_str[start:]
        print(func_str)
        return func_str

class AggregateOperator(Operator):
    def __init__(self, agenda, model_prompt, y_attr, num_samples):
        super().__init__(agenda, model_prompt, y_attr)
        self.num_samples = num_samples
    def __str__(self):
        return f"groupby-aggregation operator."
    def generate_new_feature(self, temp):
        op_prompt = aggregator_prompt_sampling.format(y_attr= self.y_attr)
        prompt = "Dataset feature description: " + self.data_agenda + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        answer = gpt_sampling(prompt = prompt, n = self.num_samples, temperature=temp)
        res_lst = []
        for i in range(len(answer)):
            try:
                res_dic = re.search(r'{(.*?)}', answer[i])
                answer_dict = eval(res_dic.group(0))
                answer_dict['new_feature'] = 'GROUPBY_' + str(answer_dict['groupby_col']) + '_' + str(answer_dict['function']) + '_' + str(answer_dict['agg_col'])
                print(answer_dict)
                res_lst.append(answer_dict)
            except:
                print("result cannot parse")
                res_lst.append(None)
        return res_lst

    
class MultiExtractor(Operator):
    def __init__(self, agenda, model_prompt, y_attr):
        super().__init__(agenda, model_prompt, y_attr)
    def __str__(self):
        return f"Multiextract operator."
    def generate_new_feature(self, temp):
        data_prompt = self.data_agenda
        op_prompt = extractor_prompt_sampling.format(y_attr= self.y_attr)
        prompt = "Dataset feature description: " + data_prompt + "Downstream machine learning models:" + self.model_prompt +'\n'+ op_prompt
        answer = gpt_sampling_extract(prompt = prompt, n = 1, temperature=temp)
        try:
            res_lst = [] 
            answer_dict = eval(answer[0])
            print(answer_dict)
            res_lst.append(answer_dict)
        except:
            print("result cannot parse")
            res_lst.append(None)
        return res_lst
    def find_function(self, result_dict, temp=0.3):
        new_col, descr, rel_cols, rel_agenda = self.parse_output(result_dict)
        prompt = extractor_function_prompt.format(new_feature= new_col, relevant= rel_cols, description= descr, rel_agenda= rel_agenda)
        # Try to generate the function
        answer = gpt_sampling_extract(prompt = prompt, n = 1, temperature=temp)[0]
        print("=========================")
        print(answer)
        print("=========================")
        if 'NEED' in answer:
            print("Need to use text completion.")
            return 'TEXT'
        elif 'Cannot' in answer:
            print("Cannot find a function or use text completion.")
            return None
        else:           
            if 'EXTERNAL' in answer:
                print("External sources needed")
                print(answer)
                return None
            else:
                pattern = r"```python(.*?)```"
                match = re.search(pattern, answer, re.DOTALL)
                code = match.group(1).strip()
                return code            
    def parse_output(self, result_dict):
        new_col = result_dict['new_feature']
        descr = result_dict['description']
        rel_cols = result_dict['relevant']
        rel_agenda = obtain_rel_agenda(rel_cols, self.data_agenda)
        return new_col, descr, rel_cols, rel_agenda