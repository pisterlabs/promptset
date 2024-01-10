# 代码描述了如何让响应对代码友好：输出解析器，这里只是简单定义了JSON格式，更强大的定义在第7章
from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd

template: str = """您是一位专业的鲜花店文案撰写员\n
对于售价为{price}元的{flower_name},您能提供一个吸引人的简短描述吗？请用中文回答
{format_instructions}
"""
# 响应解析器，将来会按指定格式解析响应，同时可以提供对该格式的描述用于模板
output_parser: StructuredOutputParser = StructuredOutputParser.from_response_schemas(
    response_schemas=[
        ResponseSchema(name="flower", description="鲜花种类"),
        ResponseSchema(name="price", description="鲜花价格", type='int'),
        ResponseSchema(name="description", description="鲜花的描述文案"),
        ResponseSchema(name="reason", description="问什么要这样写这个文案"),
    ],
)
prompt: PromptTemplate = PromptTemplate.from_template(
    template,
    # 提前填入部分变量
    partial_variables={
        'format_instructions': output_parser.get_format_instructions(),
    },
)

mode: OpenAI = OpenAI(model_name='text-davinci-003')

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

df = pd.DataFrame(columns=["flower", "price", "description", "reason"])

for f, p in zip(flowers, prices):
    _input: str = prompt.format(flower_name=f, price=p)
    print(f'input={_input}')
    _output: str = mode(_input)
    print(f'output={_output}')
    parsed_output: dict = output_parser.parse(_output)
    # parsed_output['flower'] = f
    # parsed_output['price'] = p
    df.loc[len(df)] = parsed_output

df.to_csv("flowers_with_descriptions.csv", index=False)
