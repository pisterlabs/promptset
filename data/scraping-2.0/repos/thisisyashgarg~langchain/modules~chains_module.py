from langchain.chains import LLMChain, SequentialChain
from modules.models_module import llm
from modules.prompts_module import cast_template, story_template
from modules.memory_module import story_memory, cast_memory

# Chains
cast_chain = LLMChain(llm=llm, prompt=cast_template,
                      verbose=True, output_key='cast', memory=cast_memory)
story_chain = LLMChain(llm=llm, prompt=story_template,
                       verbose=True, output_key='story', memory=story_memory)
# sequential_chain = SequentialChain(
#     chains=[cast_chain, story_chain], input_variables=['title'], output_variables=['cast', 'story'], verbose=True)
