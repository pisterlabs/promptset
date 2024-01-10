import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

model_url = "http://127.0.0.1:5552"

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from repolya.local.textgen import TextGen
# langchain.debug = True
llm = TextGen(
    model_url=model_url,
    temperature=0.01,
    top_p=0.9,
    seed=10,
    max_new_tokens=200, # 250/2500
    # stop=["\nHuman:", "\n```\n"],
    stop=[],
    streaming=False,
)


from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
)
tools = load_tools(["llm-math"], llm=llm)


from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish

def parse_json(_text):
    _res = {}
    import re
    import json
    m = re.search(r"{\n?\"?action\"?: ?\"?([^\"]*)\"?, ?\n?\"?action_input\"?: ?\"?([^\"]*)\"?\n?}", _text, re.DOTALL)
    if m is not None:
        _res["action"] = m.group(1)
        _res["action_input"] = m.group(2)
    # print(m)
    _json = json.loads(json.dumps(_res, ensure_ascii=False))
    # print(f"\n>>> text: {_text}")
    print(f">>> json: {_json}\n")
    return _res
# print(parse_json("\nAI: [JSON] {action: \"Calculator\", action_input:\"4**(2.1)\"}."))
# print(parse_json("\nAI: ```json\n{\"action\": \"Final Answer\",\n\"action_input\": \"\"}\n```"))
# print(parse_json_markdown("\nAI: ```json\n{\"action\": \"Final Answer\",\n\"action_input\": \"\"}\n```"))
# exit()

class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            # this will work IF the text is a valid JSON with action and action_input
            # response = parse_json_markdown(text)
            response = parse_json(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "conversational_chat"

# initialize output parser for agent
parser = OutputParser()


from langchain.agents import initialize_agent
# initialize agent
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=False,
    early_stopping_method="generate",
    memory=memory,
    agent_kwargs={
        "output_parser": parser,
        # "stop": ["\nObservation:"],
    }
)
# print(agent.agent.llm_chain.prompt)

sys_msg = """Human:
Bot is a expert JSON builder designed to assist with a wide range of tasks.

Bot is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Bot's communication is performed using this JSON format.

Bot can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Bot are:

- "Calculator": Useful for when you need to answer questions about math.
  - To use the calculator tool, Bot should write like so:
    ```json
    {{"action": "Calculator", "action_input": "sqrt(4)"}}
    ```

Here are some previous conversations between the Bot and User:

User: Hey how are you today?
Bot: ```json
{{"action": "Final Answer", "action_input": "I'm good thanks, how are you?"}}
```
User: I'm great, what is the square root of 4?
Bot: ```json
{{"action": "Calculator", "action_input": "sqrt(4)"}}
```
User: 2.0
Bot: ```json
{{"action": "Final Answer", "action_input": "The answer is 2"}}
```
User: Thanks could you tell me what 4 to the power of 2 is?
Bot: ```json
{{"action": "Calculator", "action_input": "4**2"}}
```
User: 16.0
Bot: ```json
{{"action": "Final Answer", "action_input": "The answer is 16.0"}}
```

Here is the latest conversation between Bot and User."""
new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt


human_msg = "Respond to the following in JSON with 'action' and 'action_input' values:\n{input}\n\nAssistant:\n"
agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg
print(agent.agent.llm_chain.prompt)


print(agent("hey how are you today?")["output"])
print(agent("what is 4 to the power of 2.1?")["output"])
print(agent("can you multiply that by 3?")["output"])

