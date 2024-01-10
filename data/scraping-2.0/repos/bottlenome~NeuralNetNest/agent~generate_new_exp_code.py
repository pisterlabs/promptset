"""
action_itemの記述に沿って元コードを改修した新しいコードを生成する
"""

import argparse
from langchain.prompts import PromptTemplate 
from langchain.llms import OpenAI
from pydantic import ValidationError


template = """
以下の元コードを改善点に基づいて修正したコードを出力してください。
返答に説明は不要です。そのまま実行可能なpythonコードの中身のみを返答してください。
元コード:
{code}

改善点:
{action_item}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--exp_code_path", required=True)
parser.add_argument("--action_item", required=True)
parser.add_argument("--llm_model", default="text-davinci-003", required=True)
args = parser.parse_args() 

exp_code_path = args.exp_code_path
action_item = args.action_item
llm_model = args.llm_model

with open(exp_code_path, 'r') as file:
    org_code = file.read()

prompt = PromptTemplate(template=template, input_variables=["code", "action_item"])
prompt_text = prompt.format(code=org_code, action_item=action_item)
print(prompt_text)
llm = OpenAI(model_name=llm_model)
new_code = llm(prompt_text)
print("####", new_code)

with open(exp_code_path, "w") as file:
    file.write(new_code)