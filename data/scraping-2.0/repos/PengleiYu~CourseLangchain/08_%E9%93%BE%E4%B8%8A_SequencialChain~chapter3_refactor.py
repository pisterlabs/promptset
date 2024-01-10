# 重构为使用chain请求
import pandas as pd
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

# 响应解析器，将来会按指定格式解析响应，同时可以提供对该格式的描述用于模板
output_parser: StructuredOutputParser = StructuredOutputParser.from_response_schemas(
    response_schemas=[
        ResponseSchema(name="flower", description="鲜花种类"),
        ResponseSchema(name="price", description="鲜花价格", type='int'),
        ResponseSchema(name="description", description="鲜花的描述文案"),
        ResponseSchema(name="reason", description="问什么要这样写这个文案"),
    ],
)
template: str = """您是一位专业的鲜花店文案撰写员\n
对于售价为{price}元的{flower_name},您能提供一个吸引人的简短描述吗？请用中文回答
{format_instructions}
"""
prompt: PromptTemplate = PromptTemplate.from_template(
    template,
    # 提前填入部分变量
    partial_variables={
        'format_instructions': output_parser.get_format_instructions(),
    },
)

chain = LLMChain(llm=OpenAI(), prompt=prompt, output_parser=output_parser)

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

df = pd.DataFrame(columns=["flower", "price", "description", "reason"])

for f, p in zip(flowers, prices):
    _output: str = chain.run(flower_name=f, price=p, )
    print(f'output={_output}')
    df.loc[len(df)] = _output

df.to_csv("flowers_with_descriptions.csv", index=False)

# print(df.to_dict())
