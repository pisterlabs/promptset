import os
import boto3
import jsonlines
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.llms.bedrock import Bedrock
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


def get_llm(model_id='amazon.titan-tg1-large'):    
    boto3_bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')

    llm = Bedrock(model_id=model_id, client=boto3_bedrock)
    return llm    

def generate_and_print(llm, q, label):
    total_prompt = """"""

    template = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when given the above statement.\n\n"""
    prompt_template = PromptTemplate(input_variables=["statement"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
    total_prompt = total_prompt + template

    template = """Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
    total_prompt = total_prompt + template

    template = """Based on the above assertions, the final response is FALSE if one of the assertions is FALSE. Otherwise, TRUE. You should only respond with TRUE or FALSE.'{}'""".format(q)
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    total_prompt = total_prompt + template

    overall_chain = SimpleSequentialChain(chains=[assumptions_chain, fact_checker_chain, answer_chain], verbose=True)
    answer = overall_chain.run(q)

    return answer

def read_questions(llm):
    file='./knowledge_qa_test.jsonl'
    with jsonlines.open(file,'r') as json_f:
        for data in json_f:
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            claims = data.get("claims", [])
            label = data.get("label", "")
            entry_point = data.get("entry_point", "")

            print("Prompt:", prompt)
            print("Response:", response)
            print("Claims:", claims)
            print("label:", label)
            print("entry_point:", entry_point)
            print("\n")
            generate_and_print(llm, response, label)

def main():
    llm = get_llm(model_id="anthropic.claude-v2")

    read_questions(llm)

if __name__ == "__main__":
    main()
