from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile

from agents.linkedin_lookup_agent import lookup as get_linkedin_profile_url

if __name__ == "__main__":
    print("Hello LangChain!")

    linkedin_profile_url = get_linkedin_profile_url(name="Lia Piovesan PUC")

    summary_template = """
        given the Linkedin {information} about a person from I want to know more about them:
        1. a short summary
        2. two interesting facts about them
        3. According her skills how she can helpe me in a supply chain project
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))
