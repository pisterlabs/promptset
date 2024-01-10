import os
from src.auth import openai_api_key, serp_api_key
from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType
from langchain.llms import OpenAI
from src.Data_Ingestion import get_articles
from src.utils import get_date
from src.logger import logging


os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPAPI_API_KEY"] = serp_api_key

def get_data():
    llm = OpenAI(temperature=0.9)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    date = get_date()

    paths = get_articles()

    logging.info("Data Ingestion Completed")

    os.makedirs("dags//src//news_letters//"+date, exist_ok=True)

    for path in paths:
        with open(path, "r") as file:
            text = file.read()

        try:
            result = agent.run(text)
        except Exception as e:
            result = ""
            
        with open(f"dags//src//news_letters//{date}//news_letter.txt", "a") as file:
            file.write(result)
            file.write("\n")
    logging.info("Langchain used")      





# version = langchain.__version__
# print(version)

# import sys
# print(sys.path)
# sys.path.append("C:\\Users\\azark\\Desktop\\News_Letter_Project\\venv")

if __name__ == "__main__":
    get_data()










