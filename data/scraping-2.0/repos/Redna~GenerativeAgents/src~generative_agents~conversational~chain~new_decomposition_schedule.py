from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Here was {agent}'s originally planned schedule from {start_hour} to {end_hour}.
{schedule}

But {agent} unexpectedly ended up {new_event} for {new_event_duration} minutes. Revise {agent}'s schedule from {start_hour} to {end_hour} accordingly (it has to end by {end_hour}).
The revised schedule:
{new_schedule_init}
"""

_prompt = PromptTemplate(input_variables=["agent",
                                            "start_hour",
                                            "end_hour",
                                            "schedule",
                                            "new_event",
                                            "new_event_duration",
                                            "new_schedule_init"
                                            ],
                         template=_template)

new_decomposition_schedule_chain = LLMChain(prompt=_prompt, llm=llm)
