import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
import numpy as np
import pandas as pd

def data_preproessing(data_df, y_label):
    # first remove the index columns
    data_df = data_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    for c in list(data_df.columns):
        if 'Unnamed' in c:
            data_df = data_df.drop(columns=[c])
    # drop NAN and inf
    data_df = data_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    # obtain the set of features
    attributes = list(data_df.columns)
    attributes.remove(y_label)
    features = attributes
    # factorize these columns
    for c in data_df.columns:
        if type(data_df[c][0]) != np.int64 and type(data_df[c][0]) != np.float64:
            print(type(data_df[c][0]))
            data_df[c] = data_df[c].astype(object)
            data_df[c], _  = pd.factorize(data_df[c])
    return data_df, features

def obtain_rel_agenda(rel_cols, agenda):
    rel_agenda = []
    for col in rel_cols:
        start = agenda.find(col)
        end = agenda.find('\n', start)
        if end == -1:
            end = None
        rel_agenda.append(agenda[start:end])
    return ', '.join(rel_agenda)


def obtain_function_name(func_str):
    match = re.search(r"def (\w+)\(", func_str)
    return match.group(1)


def obtain_relevant_cols(output_dic):
    rel_cols = output_dic["relevant"].split('\n')
    rel_cols_new = []
    for c in rel_cols:
        c = c.strip()
        rel_cols_new.append(c)
    return rel_cols_new
    

def row_serialization(row, attr_lst):
    row_ser = ''
    for a in attr_lst:
        row_ser = row_ser + str(a) + ":" + str(row[a])+ ","
    return row_ser


def obtain_function_new_features(func_str):
    pos = func_str.find('return')
    return_str = func_str[pos+7:]
    attributes = return_str.split(',')
    return attributes


def text_completion_extract(df, new_feature, temp=0.1):
    # use 3.5 turbo here for better efficiency
    llm = OpenAI(temperature = temp, model_name='gpt-3.5-turbo')
    new_col_val = []
    for idx, row in df.iterrows():
        attr_lst = list(df.columns)
        row_str = row_serialization(row, attr_lst)
        if idx == 5:
            # interact with the user to see if they want to continue
            user_input = input("Five row-completion examples have been displayed, If you want to enable this feature, type 'Continue'; otherwise, press 'Enter': ")
            if user_input != 'Continue':
                raise ValueError("The user does not want this feature") 
        try:
            response_schema = [
                ResponseSchema(name=new_feature, description="string or float, representing attribute value"),
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schema)
            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template=row_str + "{new_feature}:? \n{format_instructions}",
                input_variables=["new_feature"],
                partial_variables={"format_instructions": format_instructions}
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            result_str = chain.run({"new_feature": new_feature})
            print(result_str)
            result_dict = output_parser.parse(result_str)
            new_value = result_dict[new_feature]
            new_col_val.append(new_value)
        except Exception as e:
            print("Error", str(e))
            new_col_val.append(np.nan)
    df[new_feature] = new_col_val