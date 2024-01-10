"""
summarize_chat_relationship_v2.txt

Variables: 
!<INPUT 0>! -- Statements
!<INPUT 1>! -- curr persona name
!<INPUT 2>! -- target_persona.scratch.name

<commentblockmarker>###</commentblockmarker>
[Statements]
!<INPUT 0>!

Based on the statements above, summarize !<INPUT 1>! and !<INPUT 2>!'s relationship. What do they feel or know about each other?
"""

from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
[Statements]
{statements}

Based on the statements above, summarize {agent} and {agent_with}'s relationship. What do they feel or know about each other?
"""

_prompt = PromptTemplate(input_variables=["statements",
                                            "agent",
                                            "agent_with"],
                             template=_template)

summarize_chat_relationship_chain = LLMChain(prompt=_prompt, llm=llm)