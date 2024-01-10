from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate


_template = """
Input:
{statements}

What {number_of_insights} high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))
1."""

_prompt = PromptTemplate(template=_template,
                         input_variables=["statements",
                                          "number_of_insights"])

evidence_and_insights_chain = LLMChain(prompt=_prompt, llm=llm)
