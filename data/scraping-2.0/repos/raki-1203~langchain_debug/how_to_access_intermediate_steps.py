import os
import json

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


if __name__ == '__main__':
    llm = OpenAI(temperature=0, model_name='text-davinci-002')
    tools = load_tools(['serpapi', 'llm-math'], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                             return_intermediate_steps=True)

    response = agent({"input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"})
    print(response)

    # The actual return type is a NamedTuple for the agent action, and then an observation
    print(response['intermediate_steps'])

    print(json.dumps(response['intermediate_steps'], indent=2))


