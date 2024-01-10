from datetime import datetime

from dotenv import load_dotenv
from fastapi import APIRouter, status
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.prompts import PromptTemplate

from api.validators.business_idea import Business_Analysis, CountryName

router = APIRouter(prefix="/business-idea-generator", tags=["business-idea-generator"])
load_dotenv()
openai_llm = OpenAI(temperature=0.6)
google_repo_id = repo_id = "google/flan-t5-xxl"
current_year = datetime.now().year

google_llm = HuggingFaceHub(
    repo_id=google_repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
agent = initialize_agent(
    tools=load_tools(["wikipedia", "serpapi"], llm=openai_llm),
    llm=openai_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
prompt = PromptTemplate(
    input_variables=["country_name", "current_year"],
    template="give me a good business idea for {country_name} in {current_year}. Answer in only one word.",
)
business_chain = LLMChain(llm=google_llm, prompt=prompt, output_key="business_idea")
analysis_prompt = PromptTemplate(
    input_variables=["business_idea", "country_name"],
    template="give me an analysis of {business_idea} in {country_name}.",
)
analysis_chain = LLMChain(
    llm=google_llm, prompt=analysis_prompt, output_key="business_analysis"
)
financial_prompt = PromptTemplate(
    input_variables=["business_idea", "country_name", "current_year"],
    template="give me an financial data of {business_idea} in {country_name}. Need data untill {current_year}.",
)
financial_chain = LLMChain(
    llm=agent, prompt=financial_prompt, output_key="financial_data"
)
chain = SequentialChain(
    chains=[business_chain, analysis_chain],
    input_variables=["country_name", "current_year"],
    output_variables=["business_idea", "business_analysis"],
    verbose=True,
)


@router.post("/", status_code=status.HTTP_200_OK, response_model=Business_Analysis)
def business(country_name: CountryName) -> Business_Analysis:
    result = chain(
        {"country_name": country_name.country_name, "current_year": current_year}
    )
    result["financial_data"] = agent.run(
        financial_prompt.format(
            business_idea=result["business_idea"],
            country_name=country_name.country_name,
            current_year=current_year,
        )
    )
    return result
