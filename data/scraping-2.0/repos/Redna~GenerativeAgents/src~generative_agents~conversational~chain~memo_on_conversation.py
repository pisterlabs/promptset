"""
Variables: 
!<INPUT 0>! -- All convo utterances
!<INPUT 1>! -- persona name
!<INPUT 2>! -- persona name
!<INPUT 3>! -- persona name

<commentblockmarker>###</commentblockmarker>
[Conversation]
!<INPUT 0>!

Write down if there is anything from the conversation that !<INPUT 1>! might have found interesting from !<INPUT 2>!'s perspective, in a full sentence. 

"!<INPUT 3>!
"""

from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """
[Conversation]
{conversation}

Write down if there is anything from the conversation that {agent} might have found interesting from {agent}'s perspective, in a full sentence.

"{agent}"""

_prompt = PromptTemplate(template=_template,
                         input_variables=["conversation",
                                          "agent"])

memo_on_conversation_chain = LLMChain(prompt=_prompt, llm=llm)
