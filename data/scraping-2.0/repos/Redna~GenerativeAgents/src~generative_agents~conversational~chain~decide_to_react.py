from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Task -- given context and three options that a subject can take, determine which option is the most acceptable.

Format: 
Context: [fill in]
My question: [fill in]
Reasoning: [fill in]
Answer (in the format of Option x): [fill in]
---
Context: {context}
Right now, it is {current_time}.
{agent_observation}
{agent_with_observation}

My question: Let's think step by step. Of the following three options, what should {agent} do?
Option 1: Wait on {initial_action_description} until {agent_with} is done {agent_with_action}
Option 2: Continue on to {initial_action_description} now
"""

_prompt = PromptTemplate(input_variables=["context",
                                          "current_time",
                                          "agent", 
                                          "agent_with", 
                                          "agent_with_action",
                                          "agent_observation",
                                          "agent_with_observation",
                                          "initial_action_description",
                                          ],
                         template=_template)

decide_to_react_chain = LLMChain(prompt=_prompt, llm=llm)
