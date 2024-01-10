from langchain.chains import SequentialChain
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_plan_template = """
{statements}
Given the statements above, is there anything that {name} should remember as they plan for {today}?
If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)
Write the response from {name}'s perspective.
"""

_plan_prompt_template = PromptTemplate(
    input_variables=["statements", "name", "today"], template=_plan_template)
_plan_prompt_chain = LLMChain(
    llm=llm, prompt=_plan_prompt_template, output_key="plan_note")

_thought_template = """
{statements}
Given the statements above, how might we summarize {name}'s feelings about their days up to now?
Write the response from {name}'s perspective.
"""

_thought_prompt_template = PromptTemplate(
    input_variables=["statements", "name"], template=_thought_template)
_thought_prompt_chain = LLMChain(
    llm=llm, prompt=_thought_prompt_template, output_key="thought_note")


_currently_template = """
{name}'s status from {yesterday}:
{current_activity}

{name}'s thoughts at the end of {yesterday}:
{thought_note} {plan_note}
It is now {today}. Given the above, write {name}'s status for {today} that reflects {name}'s thoughts at the end of {yesterday}. Write this in third-person talking about {name}.
If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).
Follow this format below:
Status: <new status>
"""

_currently_prompt_template = PromptTemplate(input_variables=[
                                            "name", "yesterday", "current_activity", "thought_note", "plan_note", "today"], template=_currently_template)
_currently_prompt_chain = LLMChain(
    llm=llm, prompt=_currently_prompt_template, output_key="currently")


_daily_plan_template = """
{identity}
Today is {today}. Here is {name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).
Follow this format (the list should have 4~6 items but no more):
1. wake up and complete the morning routine at <time>, 2. ...
"""

_daily_plan_prompt_template = PromptTemplate(
    input_variables=["identity", "today", "name"], template=_daily_plan_template)
_daily_plan_prompt_chain = LLMChain(
    llm=llm, prompt=_daily_plan_prompt_template, output_key="daily_plan")

daily_plan_and_status_chain = SequentialChain(chains=[_plan_prompt_chain, _thought_prompt_chain, _currently_prompt_chain, _daily_plan_prompt_chain],
                                              input_variables=[
                                                  "name", "identity", "today", "yesterday", "statements", "today", "current_activity"],
                                              output_variables=["currently", "daily_plan"])
