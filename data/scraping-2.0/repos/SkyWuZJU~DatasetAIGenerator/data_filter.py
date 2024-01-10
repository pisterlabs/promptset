import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://xiaoma-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "44a43d50221443c690df1c2fd0f9cc14"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

from langchain.prompts import PromptTemplate
from langchain.chat_models import azure_openai
from langchain.chains import LLMChain
import json

# 失败增广案例的存储
def store_illeagle_prompts(file_name, initial_prompt, evolved_prompt, reason=[]): # reason是违法原因
    with open(file_name, 'a') as f: # 'w' is for writing from scratch, 'a' is for appending
        f.write(json.dumps(
            {'Initial Prompt': initial_prompt, 'Evolved Prompt': evolved_prompt, 'Reason': reason},
            ensure_ascii=False # ensure_ascii=False is to keep the Chinese charactors
        ) + '\n')

# 装饰器，用于检查evolved_prompt是否是异常None值
def check_none(func):
    def wrapper(*args, **kwargs):
        if kwargs['evolved_prompt'] is None:
            print('###check_none decorator: False###')
            return False
        else:
            return func(*args, **kwargs)
    return wrapper

# 过滤类型：新指令和旧指令换汤不换药，表达的是同一个问题
from prompt_templates import equal_judger

@check_none
def is_not_equal(prompt, evolved_prompt, TEMPERTATURE = 0):
    judgement_prompt = PromptTemplate(
        input_variables=['FirstPrompt', 'SecondPrompt'],
        template = equal_judger
    )
    llm = azure_openai.AzureChatOpenAI(
        deployment_name = 'turbo35',
        temperature = TEMPERTATURE,
    )
    llm_chain = LLMChain(
        prompt = judgement_prompt,
        llm = llm
    )
    judgement_output = llm_chain.predict(
        FirstPrompt = prompt,
        SecondPrompt = evolved_prompt
    )
    if judgement_output == "Not Equal" or judgement_output == "Not Equal.":
        return True
    elif judgement_output == "Equal" or judgement_output == "Equal.":
        return False
    else: # 如果判断结果不是Equal或Not Equal，则说明判断失败，则打印出来并返回False
        print('###Judgement Failure###')
        print("Error: Judgement output is neither Equal nor Not Equal.")
        print('--Initial-- \n', prompt)
        print('--Evolved--\n', evolved_prompt)
        print('--Judgement Result--\n', judgement_output)
        return False
print(is_not_equal("1+1=?", evolved_prompt="Hello what is your name")) #测试用

# 非法关键词过滤，如果生成的指令中包含非法关键词，则返回False
def no_kw(evolved_prompt):
    if evolved_prompt is None: #极端情况下，生成值是None，此时直接返回False
        return False
    # 全文不允许出现的关键词
    illegal_keywords = [
        "#Rewritten Prompt#", "rewritten prompt", "given prompt", '#Given Prompt#', '#Created Prompt#', 'created prompt'
    ]
    for keyword in illegal_keywords:
        if keyword in evolved_prompt:
            return False
    # 开头或结尾不允许出现的关键词
    illeagal_keywords_edge = ["边缘法词1", "边缘非法词2"]
    for keyword in illeagal_keywords_edge:
        if evolved_prompt.startswith(keyword) or evolved_prompt.endswith(keyword):
            return False
    # 其他情况返回True
    return True
# print(no_kw("Hello what边缘法词1 is your name边缘法词1")) #测试用


# 汇总所有类型的总过滤器
# if 所有规则都通过，则返回True
# else，返回一个反映了各项检查结果的字典{'检查项': True/False}
def prompt_filter(prompt, evolved_prompt):
    #各项检查结果
    equal_result = is_not_equal(prompt, evolved_prompt)
    kw_result = no_kw(evolved_prompt)
    if equal_result and kw_result:
        return True
    else: 
        result = {
            'Equal': equal_result,
            'KW': kw_result
        }
        return result
# print(prompt_filter("你好，我们可以交个朋友吗？", "#Rewritten Prompt#一加一等于几？")) #测试用
