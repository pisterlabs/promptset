import os
import requests
from dotenv import load_dotenv
import json

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedIn_information
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import person_intel_parser


def ice_break(name):

    gist_url = "https://gist.githubusercontent.com/anujmahajan98/81913945f066d4d79eaccd57bd4690bd/raw/898964dd113bdef0a8433cd072cf6673c9735f5e/gistfile1.txt"

    linkedin_profile_url = linkedin_lookup_agent(name=name)
    # print("LinkedIn Profile URL for Anuj Mahajan - ", linkedin_profile_url)

    summary_template = """
    Given the Linkedin information {information} about the person, I want you to create
    1. Summary of their work experience
    2. Two interesting facts about them
    3. A topic that may interest them
    4. 2 creative ice breakers to open a conversation with them.
    \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    scraped_linkedin_data = scrape_linkedIn_information(linkedin_profile_url = linkedin_profile_url)
    
    # scraped_linkedin_data = requests.get(gist_url)
    # data_dict = json.loads(scraped_linkedin_data)
    # print(type(data_dict), data_dict)

    result = chain.run(information=scraped_linkedin_data)
    print(result)
    return person_intel_parser.parse(result),scraped_linkedin_data.get(
        "profile_pic_url"
    )


if __name__ == "__main__":
    load_dotenv()
    print("Hello LangChain")
    result, profile_pic_url = ice_break(name="Anuj Mahajan")
