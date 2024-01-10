
from utils.templates_prompts import *
from langchain.chains import LLMChain
from utils.llm_utility import *



llm_critique = LLMChain(llm = small_llm , prompt=critique_prompt)
sub_task_chain = LLMChain(prompt=sub_task_prompt , llm=llm)
create_tool_experience_chain = LLMChain(llm=small_llm, prompt=missed_tool_prompt)