from langchain import PromptTemplate, LLMChain
from langchain.tools import Tool


def get_general_chain(llm):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""{question}"""
    )
    search_chain = LLMChain(llm=llm, prompt=prompt)
    search_tool = Tool(
        name='General Question',
        func=search_chain.run,
        description='General Question'
    )
    search_tool.return_direct = True
    return search_tool
