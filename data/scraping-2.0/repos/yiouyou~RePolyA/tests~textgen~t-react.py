from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.agents.agent import AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.llms import TextGen

import re
import json

import langchain
langchain.debug = True
langchain.verbose = True

from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
load_dotenv()

model_url = "http://127.0.0.1:4442"
llm = TextGen(model_url=model_url, temperature=0.01, max_new_tokens=2048)

search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events."
    )
]


template = """
[INST] <<SYS>>
Assistant is a expert JSON builder designed to assist with a wide range of tasks. Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format.

When the User has a question for the Assistant. Assistant will use tools by responding to the user with the tool use JSON format with "action" and "action_input". Tools available to Assistant are:
{tools}

To use the tool, the Assistant will write like this: 
```
{{"action": one of [{tool_names}], "action_input": a single STRING containing the information that you want to use the tool with}}
```

When the Assistant is confident of its answer to the User's question, it will respond with: ```{{"action": "Final Answer", "action_input": "Here is the answer to the User's question"}}```

<</SYS>>

User: {input}
{agent_scratchpad}
[/INST]
"""

class CustomPromptTemplate(StringPromptTemplate):
    # the template to use
    template: str
    
    # the list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        print("FORMATTING CALLED")
        intermediate_steps = kwargs.pop("intermediate_steps")
        print("INTERMEDIATE STEPS")
        print(intermediate_steps)
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
prompt = CustomPromptTemplate(template=template, tools=tools, input_variables=["input", "intermediate_steps"])

def parse_json(_text):
    result = {}
    m = re.search(r"{\n?\"?action\"?: ?\"?([^\"]*)\"?, ?\n?\"?action_input\"?: ?\"?([^\"]*)\"?\n?}", _text, re.DOTALL)
    if m is not None:
        result["action"] = m.group(1)
        result["action_input"] = m.group(2)
    return result

class ReActOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print("OUTPUT PARSER CALLED")
        print("LLM OUTPUT")
        print(llm_output)
        try:
            response = parse_json(llm_output)
            action, action_input = response["action"], response["action_input"]
            print("LLM OUTPUT PARSED")
            print(f">>> {response}")
            if action == "Final Answer":
                print("PARSING AGENT FINISH") 
                return AgentFinish(
                    return_values={"output": action_input},
                    log=llm_output
                )
            else:
                print("PARSING AGENT ACTION")
                return AgentAction(
                    tool=action,
                    tool_input=action_input,
                    log=llm_output
                )
        except Exception as e:
            print("OUTPUT PARSER ERROR")
            print(e)
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output
            )

output_parser = ReActOutputParser()


llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tools
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)



# query = "The name of the incident is Nord Stream Pipeline Explosion. List all the relevant parties of this incident INCLUDING ONLY COUNTRIES AND INTERNATIONAL ORGANIZATIONS with names in both countries and organizations appearing only once."
# response = agent_executor.run(query)
# print(response)


relevant_party = "Russia"
query = f"The name of the incident is Nord Stream Pipeline Explosion. Find any statements that {relevant_party} has made regarding the incident."
response = agent_executor.run(query)
print(response)

