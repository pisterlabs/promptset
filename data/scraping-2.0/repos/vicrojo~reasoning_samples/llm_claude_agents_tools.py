import json
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----
# os.environ["AWS_DEFAULT_REGION"] = "<REGION_NAME>"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
# os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."
# os.environ["SERPAPI_API_KEY"] = #"<YOUR_SERPAPI_API_KEY>"

bedrock_client = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

#from tools.Search import get_search_results
from tools.DDGSearch import get_search_results
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms.bedrock import Bedrock

def get_weather(location:str) -> str:
 
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock_client,
        model_kwargs={"temperature": 0.1,"top_p": 0.3, "max_tokens_to_sample": 4096},
    )
    
    template = """
    Human: Answer the following questions as best you can. You have access to the following tools:    

    Use the following instructions:
    1. Question: the input question you must answer
    2. Thought: you should always think about what to do
    3. Action: the action to take, should be use a search query
    4. Action Input: the input to the action
    5. Observation: the result of the action
    6. Thought: I now know the final answer
    7. Final Answer: the final answer to the original input question
    The steps 1 to 5 can repeat N times

    Return only your answer to the human question exclusively as a JSON object in form of key:value pairs. The final answer must be named with the key "FinalAnswer", and break it down into the different names and values that compose it, in the form of key-value pairs using the following keys: Location(string), Temperature (numeric), Units(string), Conditions(string).

    What is the weather in {place} right now?
    Use the results you got from the search engine to answer the question, and don't invent a number.

    Assistant:
    """
    tools = [
        Tool (
            name="Search",
            func=get_search_results,
            description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
        )
    ]

    react_agent = initialize_agent(tools, 
                                llm, 
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                verbose=True,
                                #    max_iteration=2,
                                #    return_intermediate_steps=True,
                                handle_parsing_errors="""return the final answer as is"""
                                )    

    prompt_template = PromptTemplate(
        input_variables=["place"], template=template
    )

    weather = react_agent.run(prompt_template.format_prompt(place=location))

    return weather

weather = get_weather("Las Vegas")

print_ww(weather)