# from langchain import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from dotenv import load_dotenv
from agents.linkedIn_lookup_agent import lookup as lookup_linkedIn_agent
from output_parser.output_parser import person_intel_parser, PersonIntel
import os

load_dotenv()


def ice_breaker() -> PersonIntel:
    summary_template = """
    given the linkedIn information {information} about a person from I want you to create:
    1. a short summary
    2. the interesting facts about them
    3. a topic tha may interest 
    4. 2 ice breakers to open conversation
    \n{format_instructions}
  """
    summary_template_information = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_template_information)

    print("searching linked in profile url")

    linkedIn_profile_url = lookup_linkedIn_agent(name="Vincenzo Roberti")

    print(linkedIn_profile_url)
    linkedIn_information = scrape_linkedin_profile(
        linkedin_profile_url=linkedIn_profile_url,
        fake=False,
    )

    result = chain.run(information=linkedIn_information)
    return person_intel_parser.parse(result)


if __name__ == "__main__":
    output: PersonIntel = ice_breaker()
    print(output)
