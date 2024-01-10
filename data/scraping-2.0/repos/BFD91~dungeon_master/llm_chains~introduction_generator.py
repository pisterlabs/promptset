from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory, ConversationEntityMemory

from llm_chains.helpers.llms import gpt_3_5



introduction_generation_template = """Consider the following DnD adventure:

{adventure}

The following is a highly detailed introduction to the setting where the adventure takes place. The introduction \
consists of  a description of the area in which the adventure takes place, including geography, climate, history, political \
situation, and culture. 

"""

introduction_generation_prompt = PromptTemplate(
    input_variables=["adventure"],
    template=introduction_generation_template
)

introduction_chain = LLMChain(llm=gpt_3_5, prompt=introduction_generation_prompt)

