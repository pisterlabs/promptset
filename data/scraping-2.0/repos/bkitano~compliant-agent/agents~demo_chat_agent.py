from langchain.agents import ConversationalAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from agents.CompliantAgentExecutor import CompliantAgentExecutor
from commons import chat_model, llm_model
from langchain.agents import Tool
from vectordb.retrieval_qa import regulations_tool

"""
This is a custom conversational chat agent. We are going to
modify the prefix and agent format instructions to ensure that 
it assesses the compliance of a proposed action.
"""

PREFIX = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant is also responsible for ensuring that all of its actions comply with regulations. Assistant is capable of navigating the complexities of various industries and providing comprehensive guidance to users seeking compliance support.

Assistant adopts a proactive approach in helping users by anticipating potential compliance challenges and offering preventive measures. In addition, if Assistant identifies that certain actions requested by the user may be non-compliant, Assistant will exit the conversation and provide a reason for its non-compliance. 

If the user provides an alternative, the Assistant will try checking the compliance of the new solution, and if it is compliant, it will continue generating actions and executing those actions. If the new solution is also non-compliant, the Assistant will exit the conversation and provide a reason for its non-compliance. 

TOOLS:
------

Assistant has access to the following tools:"""

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? (Yes|No)
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I can now respond to the user with the completed action or observation.
Final Response: the final response to the original input message
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```"""


def toggle(item):
    if "alcohol" in item:
        return """<response>
        <compliance_status>FALSE</compliance_status>
        <reason>Alcohol is not allowed.</reason>
    </response>"""
    else:
        return """<response>
        <compliance_status>TRUE</compliance_status>
    </response>"""


mock_compliance_tool = Tool(
    name="Compliance Agent",
    func=toggle,
    description="""Useful for when you need to check the legality or compliance of a specific action. The tool takes in a proposed action and returns whether or not the action is compliant.

    The format of the output is:
    <response>
        <compliance_status>TRUE|FALSE</compliance_status>
        <reason>REASON</reason>
    </response>

    If an action's compliance status is FALSE, it requires the agent to use the "Exit Tool", and pass the reason to the Exit Tool as to which rules the action is in violation of.""",
    verbose=True,
)

mock_amazon_tool = Tool(
    name="Amazon Tool",
    func=lambda x: f"Successfully purchased a gift on Amazon: {x}",
    description="Useful for when you need to buy a gift on Amazon. The input should be the name of the gift. It will send out an order request for that gift to Amazon.",
    verbose=True,
    return_direct=True,
)

exit_tool = Tool(
    name="Exit Tool",
    func=lambda x: f"Exiting...{x}",
    description="Useful for when you need to exit the conversation. This tool will exit the conversation and return the reason for exiting.",
    return_direct=True,
    verbose=True,
)

agent = ConversationalAgent.from_llm_and_tools(
    llm=llm_model,
    tools=[mock_amazon_tool],
    prefix=PREFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "chat_history"],
    verbose=True,
)
loaded_mem = ConversationBufferMemory(memory_key="chat_history")

agent_executor = CompliantAgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[mock_amazon_tool],
    memory=loaded_mem,
    verbose=True,
)
