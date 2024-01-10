from y2020_tool import y2020_search
from y2021_tool import y2021_search
from y2022_tool import y2022_search
from y2023_tool import y2023_search
from y2020_2023_tool import y2020to2023_search
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

tools = [
    Tool(
        name = "2020年台大版問答",
        func=y2020_search,
        description="當你需要回答2020年的台大相關資訊的問題時，這個工具會對你很有幫助。"
    ),
    Tool(
        name = "2021年台大版問答",
        func=y2021_search,
        description="當你需要回答2021年的台大相關資訊的問題時，這個工具會對你很有幫助。"
    ),
    Tool(
        name = "2022年台大版問答",
        func=y2022_search,
        description="當你需要回答2022年的台大相關資訊的問題時，這個工具會對你很有幫助。"
    ),
    Tool(
        name = "2023年台大版問答",
        func=y2023_search,
        description="當你需要回答2023年的台大相關資訊的問題時，這個工具會對你很有幫助。"
    ),
    Tool(
        name = "台大版問答",
        func=y2020to2023_search,
        description="當你需要沒有明確表達年份的台大相關資訊的問題時，這個工具會對你很有幫助。"
    )
]

# Set up the base template
template_with_history = """請依據問題，以及「ptt 台大版」的資料給予回答。
若問題是選擇題，請擇一回答，不能有多個選擇，並提供你的論點、解釋你為什麼會選擇那個答案。

回答問題前，思考的過程中，您必須選擇以下的tools來幫助你思考：

{tools}

請使用以下格式回答:

Question: 你必須回答的輸入問題
Thought: 你該用什麼方法來解決的思考過程
Action: 你該採取的action，必須只能是 [{tool_names}]中的一個工具
Action Input: 輸入action
Observation: action後的結果
Thought: 我現在知道最終答案了
Final Answer: 原始輸入問題的最終答案

開始! 記得你必須以蒐集到的資料回答

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and tool_names variables because those are generated dynamically
    # This includes the intermediate_steps variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)


#output parser
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single output key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()