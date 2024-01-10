from langchain import OpenAI
from langchain.chains import LLMChain


def test_prompt():
    # 使用api2d访问openai
    OpenAI.openai_api_base = "https://oa.api2d.net"

    ai = OpenAI()
    ai.openai_api_base = "https://oa.api2d.net/v1"

    from generation_redbook_title import PROMPT

    chain = LLMChain(llm=ai, prompt=PROMPT)

    # data = chain.run(theme="雅诗兰黛小棕瓶", keywords=["恐龙抗狼, 酱香"])

    print(data)
