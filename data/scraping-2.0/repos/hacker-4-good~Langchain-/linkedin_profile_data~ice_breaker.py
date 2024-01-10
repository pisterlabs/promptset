from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedln import scrape_linkedln_profile
from agents.linkedln_lookup_agent import lookup as linkedin_lookup_agent
from output_parser import person_intel_parser


def ice_break(name: str):
    linkedin_profile_url = linkedin_lookup_agent(name=name)

    linkedin_profile_url = linkedin_profile_url.split(" ")[-1]

    linkedin_profile_url = linkedin_profile_url.split("/")[-1][:-1]

    linkedin_profile_url = "https://www.linkedin.com/in/" + linkedin_profile_url

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. three interesting facts about them
        3. Topic that may interest them
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedln_profile(linkedin_profile_url=linkedin_profile_url)

    result = chain.run(information=linkedin_data)

    return person_intel_parser.parse(result), linkedin_data.get('profile_pic_url')


if __name__ == "__main__":
    print("Hello Langchain!")

    result = ice_break(name="Mayank Goswami Dronacharya College of Engineering")
    print(result)
