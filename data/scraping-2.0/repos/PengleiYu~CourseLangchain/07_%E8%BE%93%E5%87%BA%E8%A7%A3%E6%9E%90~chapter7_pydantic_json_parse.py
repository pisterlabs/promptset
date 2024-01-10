# 本章类似第3章，本章使用了pydantic来定义JSON，功能更强大，更适合编程
from langchain.llms.openai import OpenAI

model = OpenAI(model_name='text-davinci-003')

import pandas as pd

df = pd.DataFrame(columns=["flower_type", "price", "description", "reason"])
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

from pydantic import BaseModel, Field


class FlowerDescription(BaseModel):
    flower_type: str = Field(description='鲜花种类')
    price: int = Field(description='鲜花价格')
    description: str = Field(description='鲜花描述文案')
    reason: str = Field(description='为什么要写这个文案')


from langchain.output_parsers.pydantic import PydanticOutputParser

output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)
format_instructions = output_parser.get_format_instructions()

from langchain.prompts.prompt import PromptTemplate

str_template = """
您是一位专业的鲜花店文案撰写员。对于售价为 {price} 元的 {flower} ，
您能提供一个吸引人的简短中文描述吗？
{format_instructions}
"""
prompt_template = PromptTemplate.from_template(
    str_template,
    partial_variables={'format_instructions': format_instructions, },
)

for f, p in zip(flowers, prices):
    _input = prompt_template.format(price=p, flower=f)
    print(_input)
    _output = model(_input)
    print(_output)
    parsed_output = output_parser.parse(_output)
    df.loc[len(df)] = parsed_output.model_dump()

# print('输出数据：', df.to_dict(orient='records'))
df.to_csv("flowers_with_descriptions.csv", index=False)
