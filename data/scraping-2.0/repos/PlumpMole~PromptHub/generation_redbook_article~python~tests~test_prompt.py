from langchain import OpenAI
from langchain.chains import LLMChain


def test_prompt():
    # 使用api2d访问openai
    OpenAI.openai_api_base = "https://oa.api2d.net"

    ai = OpenAI()
    ai.openai_api_base = "https://oa.api2d.net/v1"
    ai.model_name = "gpt-3.5-turbo-16k"

    from generation_redbook_article import PROMPT

    chain = LLMChain(llm=ai, prompt=PROMPT)

    data = chain.run(topic="安利吴恩达Prompt教程")
    print(data)
    with open("article.md", "w+") as f:
        f.write(data)
