import pandas as pd
import toml

from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage


with open("secrets.toml", "r") as file:
    secrets = toml.load(file)

openai_api_key = secrets["openai_api_key"]
llm_model = secrets["llm_model"]
langchain_verbose = str(secrets["langchain_verbose"])


def func_calling_chain():
    func_calling_json_schema = {
        "title": "get_structured_sources_to_list",
        "description": "Extract the source information from text and return it in a json.",
        "type": "object",
        "properties": {
            "sources": {
                "title": "sources",
                "description": """All sources information extracted from text, in a json format. Each source should be in a json format, including its title, publisher, the year of publication, and other information if available. The JSON should look like this:{"sources": [{"title": "平成12年工業統計調査","publisher": "経済産業省","year": 2002},{"title": "素形材年鑑","publisher": "財団法人素形材センター"}]}""",
                "type": "string",
            },
        },
        "required": ["sources"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content="You are a world class algorithm for extracting the sources information from text. Make sure to answer in the correct structured format."
        ),
        HumanMessage(content="Text:"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

    llm_func_calling = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=llm_model,
        temperature=0,
        streaming=False,
    )

    func_calling_chain = create_structured_output_chain(
        output_schema=func_calling_json_schema,
        llm=llm_func_calling,
        prompt=prompt_func_calling,
        verbose=langchain_verbose,
    )

    return func_calling_chain


# 使用pandas读取Excel文件
df = pd.read_excel("data/idea-348.xlsx")
general_comment = df["GeneralComment"]
# global_ids = df["GlobalId"]

sources_function_calling_chain = func_calling_chain()

# 新增一个空的 Response 列
df['Sources'] = pd.Series(dtype='object')
for index, comment in general_comment.items():
    try:
        func_calling_response = sources_function_calling_chain.run(comment)
        df.at[index, 'Sources'] = func_calling_response
        print(func_calling_response)
    except Exception as e:
        print(f"Error processing comment at index {index}: {e}")
        df.at[index, 'Sources'] = f"Error: {e}"

# 将修改后的 DataFrame 写回原 Excel 文件
df.to_excel("data/idea-348.xlsx", index=False)