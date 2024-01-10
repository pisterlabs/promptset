from langchain.llms import VertexAI
from tools.tools import get_google_search, get_sql_database,get_scrape_linkedin_profile, get_next_available_date, get_salary_data, get_what_day_is_today, get_recruiter_email_template, get_interview_feedback
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import GmailToolkit

from dotenv import load_dotenv
import os
load_dotenv()
if __name__ == "__main__":
    pass



gmail_toolkit = GmailToolkit()
google_search = Tool(
    name="GoogleSearch",
    func=get_google_search,
    description="useful for when you need get a google search result",
)
sql_database = Tool(
    name="SQLDatabase_data_retrieval",
    func=get_sql_database,
    description="useful for when you need to query a database for candidates. The table is called scouted_candidates. Fields are: candidate_id, first_name, last_name, hire_date, min_salary, max_salary, email, phone_number, location, experience_years, linkedin_url, notes",
)
recruiter_email_template = Tool(
    name="RecruiterEmailTemplate",
    func=get_recruiter_email_template,
    description="useful for when you need to get a recruiter email template in the final answer",
)
scrape_linkedin_profile= Tool(
    name="scrape_linkedin_profile",
    func=get_scrape_linkedin_profile,
    description="useful for getting information on a Linkedin profile url",
)
next_available_date= Tool(
    name="next_available_date",
    func=get_next_available_date,
    description="use this tool to get the next available date for an interview",
)
salary_data = Tool(
    name="get_salary_data",
    func=get_salary_data,
    description="useful for getting the salary data for a candidate on a specific location and role",
)
what_day_is_today = Tool(
    name="what_day_is_today",
    func=get_what_day_is_today,
    description="use this tool to current day today"
)
feedback_candidate = Tool(
    name="feedback_candidate",
    func=get_interview_feedback,
    description="this tool is a retiever to get the feedback of a candidate"
)

def getLLM(temperture, model):
    llm_type = os.getenv("LLM_TYPE")

    if model == "gemini-pro":
            try:
                llm = VertexAI(temperature=temperture, verbose=True, max_output_tokens=8192,model_name="gemini-pro")
            except Exception as e:
                print(str(e))
                print("Model gemini failed not found, using text-bison@002")
                llm = VertexAI(temperature=temperture, verbose=True, max_output_tokens=1020,model_name="text-bison@002")
    elif model != "":
        llm = VertexAI(temperature=temperture, verbose=True, max_output_tokens=1020,model_name=model)
    elif llm_type == "vertexai":
        llm = VertexAI(temperature=temperture, verbose=True, max_output_tokens=1020,model_name=os.getenv("VERTEX_MODEL"))
    print("\U0001F916 Model used for this agent: " + llm.model_name)
    return llm

gmail_tools = [gmail_toolkit.get_tools()[0]]

def get_gmail_agent(temperture=0) -> AgentExecutor:
    #print(f"Temperature: {temperture}")
    print("*" * 79)
    print("AGENT: Recruiter Email Crafter Agent!")
    print("*" * 79)
    llm = getLLM(temperture,os.getenv("VERTEX_MODEL_EMAIL"))
    agent = initialize_agent(
        gmail_tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent

def get_search_agent(temperture=0) -> AgentExecutor:
    #print(f"Temperature: {temperture}")
    print("*" * 79)
    print("AGENT: Recruiter information retrieval Agent!")
    print("*" * 79)
    llm = getLLM(temperture,os.getenv("VERTEX_MODEL_GATHERING"))
    tools_for_agent = [
        google_search,
        sql_database,
        scrape_linkedin_profile,
        salary_data,
        feedback_candidate
    ]

    agent = initialize_agent(
        tools_for_agent,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent


def get_salary_decision_agent(temperture=0)-> AgentExecutor:
    print("*" * 79)
    print("AGENT: HR salary decision Agent!")
    print("*" * 79)
    #print(f"Temperature: {temperature}")
   
    llm = getLLM(temperture,os.getenv("VERTEX_MODEL_SALARY"))
    #llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools_for_agent = [
        google_search,
        scrape_linkedin_profile
    ]

    agent = initialize_agent(
        tools_for_agent,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent