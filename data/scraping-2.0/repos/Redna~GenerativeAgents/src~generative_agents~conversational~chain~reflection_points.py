"""!<INPUT 0>!

Given only the information above, what are !<INPUT 1>! most salient high-level questions we can answer about the subjects in the statements?
1)
"""

from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """
{memory}

Given only the information above, what are {count} most salient high-level questions we can answer about the subjects in the statements?
"""

_prompt = PromptTemplate(template=_template,
                         input_variables=["memory",
                                          "count"])

reflection_points_chain = LLMChain(prompt=_prompt, llm=llm)
