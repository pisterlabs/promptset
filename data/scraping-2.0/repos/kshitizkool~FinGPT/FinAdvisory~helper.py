from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import get_api_key
openapi_key = get_api_key()
if openapi_key:
    # Make API requests or perform other actions using the API key
    pass
else:
    print("API key not set.")
# import os
# os.environ['OPENAI_API_KEY'] = openapi_key
llm = OpenAI(temperature=0.7)


def generate_advisory(securities):
    # Chain 1: Securities Name
    prompt_template_name = PromptTemplate(
        input_variables=['securities'],
        template='I want to invest my money in {securities}, Suggest a good way to generate maximum returns'
    )

    advisory_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="brokers")

    # Chain 2: Brokerage options
    prompt_template_name = PromptTemplate(
        input_variables=['brokers'],
        template="""Suggest some good {brokers} for this investment in india. Return it as a comma seperated string"""
    )

    advisory_items_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="fund_name")
    chain = SequentialChain(
        chains=[advisory_chain, advisory_items_chain],
        input_variables=['securities'],
        output_variables=['brokers', 'fund_name']
    )
    response = chain(securities)

    return response


if __name__ == "__main__":
    print(generate_advisory("Stocks"))
