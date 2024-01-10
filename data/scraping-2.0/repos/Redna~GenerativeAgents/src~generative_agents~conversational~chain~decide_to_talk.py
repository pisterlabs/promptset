from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Task -- given context, determine whether the subject will initiate a conversation with another.
Format:
Context: []
Question: []
Reasoning: []
Answer in "yes" or "no": []
---
Context: {context}
Right now, it is {current_time}. {init_agent} and {agent_with} {last_chat_summary}.
{init_agent_observation}
{agent_with_observation}

Question: Would {init_agent} initiate a conversation with {agent_with}?

Reasoning: Let's think step by step.
"""

_prompt = PromptTemplate(input_variables=["context",
                                          "current_time",
                                          "init_agent", 
                                          "agent_with", 
                                          "last_chat_summary",  
                                          "init_agent_observation", 
                                          "agent_with_observation"],
                         template=_template)

decide_to_talk_chain = LLMChain(prompt=_prompt, llm=llm)
