from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from custom_tools import QASearchTool, HumanTool, GoogleSerperTool, GeneralKnowledgeTool

PREFIX = """You are a helpful and honest physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC).
If you do not know the answer, you will not make up an answer and instead ask further questions for clarifications

Your job is to ANALYSE the given patient profile based on given query based on one of the following criteria:
- Whether treated patient is new patient or patient under maintenance
- Prior response to Infliximab
- Prior failure to Anti-TNF agents
- Prior failure to Vedolizumab
- Age
- Pregnancy
- Extraintestinale manifestations
- Pouchitis

FINALLY RETURN up to 2 TOP choices of biological drugs given patient profile. Explain the PROS and CONS of the 2 choices.

You have access to the following tools:"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SUFFIX = """Begin!

Patient Profile: {input}
Thought: {agent_scratchpad}"""

if __name__ == "__main__":

    PROMPT_TEMPLATE = ZeroShotAgent.create_prompt(
        tools=[],
        prefix=PREFIX,
        suffix=SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,    
    )

    print(PROMPT_TEMPLATE.template)

