# !pip install langchain==0.0.292

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI()  # 默认是text-davinci-003模型
llm.predict("你好，欢迎")

chat_model = ChatOpenAI()  # 默认是gpt-3.5-turbo
chat_model.predict("你好，欢迎")

# 多轮对话封装
from langchain.schema import (
    AIMessage, #等价于OpenAI接口中的assistant role
    HumanMessage, #等价于OpenAI接口中的user role
    SystemMessage #等价于OpenAI接口中的system role
)

messages = [
    SystemMessage(content="你是AGIClass的课程助理。"),
    HumanMessage(content="我来上课了")
]
chat_model(messages)

# tempelate 封装
# chatTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是{product}的客服助手。你的名字叫{name}"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

llm = ChatOpenAI()
llm(
    template.format_messages(
        product="AGI课堂",
        name="瓜瓜",
        query="你是谁"
    )
)

# fewshot Template

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

#例子(few-shot)
examples = [
    {
        "input": "北京天气怎么样",
        "output" : "北京市"
    },
    {
        "input": "南京下雨吗",
        "output" : "南京市"
    },
    {
        "input": "江城热吗",
        "output" : "武汉市"
    }
]

#例子拼装的格式
example_prompt = PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}")

#Prompt模板
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

prompt = prompt.format(input="羊城多少度")

print("===Prompt===")
print(prompt)

llm = OpenAI()
response = llm(prompt)

print("===Response===")
print(response)

# Output Parser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Dict
import json

# 避免print时中文变成unicode码
def chinese_friendly(string):
    lines = string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return '\n'.join(lines)


model_name = 'gpt-4'
temperature = 0
model = OpenAI(model_name=model_name, temperature=temperature)

# 定义你的输出格式
class Command(BaseModel):
    command: str = Field(description="linux shell命令名")
    arguments: Dict[str, str] = Field(description="命令的参数 (name:value)")

    # 你可以添加自定义的校验机制
    @field_validator('command')
    def no_space(cls, field):
        if " " in field or "\t" in field or "\n" in field:
            raise ValueError("命令名中不能包含空格或回车!")
        return field

# 根据Pydantic对象的定义，构造一个OutputParser
parser = PydanticOutputParser(pydantic_object=Command)

prompt = PromptTemplate(
    template="将用户的指令转换成linux命令.\n{format_instructions}\n{query}",
    input_variables=["query"],
    # 直接从OutputParser中获取输出描述，并对模板的变量预先赋值
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

print("====Format Instruction=====")
print(chinese_friendly(parser.get_format_instructions()))
print(parser.get_format_instructions())

query = "将系统日期设为2023-04-01"
model_input = prompt.format_prompt(query=query)

print("====Prompt=====")
print(chinese_friendly(model_input.to_string()))

output = model(model_input.to_string())
print("====Output=====")
print(output)
print("====Parsed=====")
cmd = parser.parse(output)
print(cmd)

# Auto-fixing parser
from langchain.output_parsers import OutputFixingParser

new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(model="gpt-4"))

# 我们把之前output的格式改错
output = output.replace("\"", "'")
print("===格式错误的Output===")
print(output)
try:
    cmd = parser.parse(output)
except Exception as e:
    print("===出现异常===")
    print(e)

# 用OutputFixingParser自动修复并解析
cmd = new_parser.parse(output)
print("===重新解析结果===")
print(cmd)







