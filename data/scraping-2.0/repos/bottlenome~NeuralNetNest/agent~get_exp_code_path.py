"""
summaryからexp_code_pathを取得する
"""

import argparse
from langchain.prompts import PromptTemplate 
from langchain.llms import OpenAI
from command_util import run_command


template = """
下記のissue summaryとディレクトリ構成を元に編集すべきコードのパスを示してください。   
返答に説明は不要です。コードのパスのみ示してください。
issue summary:
{issue_summary}
pwd結果:
{pwd_result}
ls結果:
{ls_reuslt}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--issue_summary", required=True)
parser.add_argument("--llm_model", default="text-davinci-003")
args = parser.parse_args() 

issue_summary = args.issue_summary
llm_model = args.llm_model
ls_reuslt = run_command("ls", capture_output=True)
pwd_result = run_command("pwd", capture_output=True)

prompt = PromptTemplate(template=template, input_variables=["issue_summary", 'pwd_result', 'ls_reuslt'])
prompt_text = prompt.format(issue_summary=issue_summary, pwd_result=pwd_result, ls_reuslt=ls_reuslt)
llm = OpenAI(model_name=llm_model)
code_path = llm(prompt_text)

print(code_path)