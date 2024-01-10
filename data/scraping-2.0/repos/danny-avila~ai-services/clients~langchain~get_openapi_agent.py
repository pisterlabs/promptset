import json
import yaml
from langchain.agents import create_openapi_agent
from langchain.agents.agent_toolkits import OpenAPIToolkit
from langchain.llms.openai import OpenAI
from langchain.requests import RequestsWrapper
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec

# def convert_json_to_yaml(json_file, yaml_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)

#     with open(yaml_file, 'w') as f:
#         yaml.dump(data, f, default_flow_style=False)

def get_openapi_agent(api_key, model_name, plugin_name):
    # with open(f"{plugin_name}_openapi.json") as f:
    #     data = json.load(f)
    # json_spec=JsonSpec(dict_=data)
    # convert_json_to_yaml(f"{plugin_name}_openapi.json", f"{plugin_name}_openapi.yaml")

    # with open(f"{plugin_name}_openapi.yaml") as f:
    #     raw_api_spec  = yaml.load(f, Loader=yaml.FullLoader)
    # json_spec=JsonSpec(dict_=data, max_value_length=4000)

    with open(f"{plugin_name}_openapi.yaml") as f:
        raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
    openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)

    headers = {
        "Accept": "application/json"
    }

    # openai_requests_wrapper=RequestsWrapper(headers=headers)
    requests_wrapper=RequestsWrapper(headers=headers)
    llm=OpenAI(openai_api_key=api_key, model_name=model_name, temperature=0)

    # openapi_toolkit = OpenAPIToolkit.from_llm(OpenAI(openai_api_key=api_key, model_name=model_name, temperature=0), json_spec, openai_requests_wrapper, verbose=True)
    # openapi_agent_executor = planner.create_openapi_agent(api_spec,
    #     llm=OpenAI(openai_api_key=api_key, model_name=model_name, temperature=0),
    #     toolkit=openapi_toolkit,
    #     verbose=True
    # )


    # return openapi_agent_executor

    agent = planner.create_openapi_agent(openai_api_spec, requests_wrapper, llm)
    return agent