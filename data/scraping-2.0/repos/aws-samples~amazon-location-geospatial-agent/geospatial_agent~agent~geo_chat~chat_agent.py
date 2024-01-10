from typing import Sequence

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.tools import BaseTool
from pydispatch import dispatcher

from geospatial_agent.agent.geo_chat.tools.geocode_tool import geocode_tool
from geospatial_agent.agent.geo_chat.tools.gis_work_tool import gis_work_tool
from geospatial_agent.agent.shared import AgentSignal, EventType, SIGNAL_GEO_CHAT_INITIATED, \
    SENDER_GEO_CHAT_AGENT, SIGNAL_GEO_CHAT_RESPONSE_COMPLETE
from geospatial_agent.shared.bedrock import get_claude_v2
from geospatial_agent.shared.prompts import HUMAN_ROLE, ASSISTANT_ROLE

_PREFIX = f"""\
{HUMAN_ROLE}:
You are an conversational agent named Agent Smith. You are created by Amazon Location Service to assist an user with geospatial information and queries.
Answer the following questions as best you can.
1. To assist you, you have access to some tools. Each tool has a description that explains its functionality, the inputs it takes, and the outputs it provides.
2. You MUST answer all intermediate steps with the following prefixes: Thought, Action, Action Input, and Observation.
3. If you do not find a tool to use, you MUST add "Observation:" prefix to the output.
"""

_SUFFIX = f"""\
{HUMAN_ROLE}:
Question: {{input}}


{ASSISTANT_ROLE}:
Thought:{{agent_scratchpad}}
"""

_FORMAT_INSTRUCTIONS = f"""Use the following format:
{HUMAN_ROLE}:
Question: The input question or query you MUST answer.
Thought: You MUST always think about what to do.
Action: You SHOULD select an action to take. Actions can be one of [{{tool_names}}]. If no tool is selected, keep conversing with the user.
Action Input: The input to the action.
Observation: The output or result of the action. (this Thought/Action/Action Input/Observation can repeat N times).
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of using a tool:
{HUMAN_ROLE}:
Question: I want to know the latitude and longitude of English Bay, Vancouver.
{ASSISTANT_ROLE}:
Thought: I should use find a tool from [{{tool_names}}]. The selected tool is geocode.
Action: Geocode.
Action Input: English Bay, Vancouver.
Observation: The longitude is 49.28696 and latitude is -123.1432.
Thought: I now know the latitude and longitude of English Bay, Vancouver.
Final Answer: The longitude and latitude of English Bay, Vancouver are 49.28696 and -123.1432.

Example of not finding a tool:
{HUMAN_ROLE}:
Question: Hello, how are you?
{ASSISTANT_ROLE}:
Thought: I should greet the user in response.
Action: I do not have any tool to use here. Keep conversing with the user.
Action Input: Hello, how are you?
Observation: Hello, I am doing fine! If you have a geospatial query or action, I can help you with that.
Thought: I now have the final response for the user.
Final Answer: Hello, I am doing fine! If you have a geospatial query or action, I can help you with that.
"""


class GeoChatAgent:
    """
    GeoChatAgent class is the gateway to Amazon Location Geospatial Agent
    This class decides whether the GeospatialAgent should be invoked or not.
    If no, then it converses with the customer with the help of some tools.
    """

    def __init__(self, memory=None):
        self.memory = memory
        self.claude_v2 = get_claude_v2()

    def invoke(self, agent_input: str, storage_mode: str, session_id: str) -> str:
        tools: Sequence[BaseTool] = [geocode_tool(), gis_work_tool(session_id=session_id, storage_mode=storage_mode)]
        agent = ZeroShotAgent.from_llm_and_tools(
            llm=self.claude_v2, tools=tools,
            prefix=_PREFIX, suffix=_SUFFIX, input_variables=["input", "agent_scratchpad"],
            format_instructions=_FORMAT_INSTRUCTIONS, memory=self.memory)

        dispatcher.send(signal=SIGNAL_GEO_CHAT_INITIATED,
                        sender=SENDER_GEO_CHAT_AGENT,
                        event_data=AgentSignal(
                            event_source=SENDER_GEO_CHAT_AGENT,
                            event_message="Initiating Agent Smith, your conversational geospatial agent",
                        ))

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=self.memory)
        output = agent_executor.run(input=agent_input, chat_history=[])
        dispatcher.send(signal=SIGNAL_GEO_CHAT_RESPONSE_COMPLETE,
                        sender=SENDER_GEO_CHAT_AGENT,
                        event_data=AgentSignal(
                            event_source=SENDER_GEO_CHAT_AGENT,
                            event_type=EventType.Message,
                            event_message=output,
                            is_final=True,
                        ))
        return output
