from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.requests import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.openapi import planner
import yaml
from dotenv import dotenv_values

with open("data_files/swagger.yml") as f:
    raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)

openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
config = dotenv_values(".env")
strava_token = config.get('STRAVA_TOKEN')
headers = {
    "Authorization": "Bearer {strava_token}".format(strava_token = strava_token),
    "Content-Type": "application/json"
}
# Get API credentials.
requests_wrapper = RequestsWrapper(headers=headers)

llm = ChatOpenAI(openai_api_key=config.get('OPENAI_API_KEY'), model_name="gpt-4", temperature=0.0)
openai_agent = planner.create_openapi_agent(openai_api_spec, requests_wrapper, llm)

query = "how far was her last activity of type run? how many minutes did it take her?"
openai_agent.run(query) #it knows last is most recent
num_runs = openai_agent.run("how many runs has Lizzie done")
openai_agent.run("how many runs has Lizzie done") #119 true
openai_agent.run("how many workouts of type ride has she done?") #248 -- get 0 if i ask "how many rides"
openai_agent.run("How many swims has she done?") # 5 correct
openai_agent.run("What was Lizzie's fastest half marathon run?") # "First hawk hill run!" with a time of 12171 seconds, not the fastest #2nd was ride for yes on j
openai_agent.run("What was her fastest run of at least 21.0975km?") #ride for yes on j
openai_agent.run("How many walks with total elevation gain over 100 has she taken?") 
