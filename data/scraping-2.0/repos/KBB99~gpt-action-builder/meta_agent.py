from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from prompts import get_meta_prompt_template
from config import CONSTRAINTS, TIPS_TEMPLATE
from parsing import get_new_instructions

def initialize_meta_chain():
    """Initializes and returns a language learning model chain."""

    meta_prompt = PromptTemplate(
        input_variables=["goal", "david_instantiation_prompt", "david_execution", "actions"], 
        template=get_meta_prompt_template()
    )

    meta_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"), 
        prompt=meta_prompt, 
        verbose=True, 
    )
    return meta_chain

def update_meta_chain_and_variables(extracted_actions_output, goal, david_instantiation_prompt, execution_output):
    meta_chain = initialize_meta_chain()
    temp_prompt = PromptTemplate(
        input_variables=["tool_names","tools","input","constraints","tips","agent_scratchpad"],
        template=david_instantiation_prompt
    )
    temp_prompt = temp_prompt.format(
        tools="Bash", 
        tool_names="Bash Tool", 
        input=goal, 
        constraints=CONSTRAINTS, 
        tips=TIPS_TEMPLATE, 
        agent_scratchpad=""
    )
    meta_output = meta_chain.predict(
        goal=goal, 
        david_instantiation_prompt=temp_prompt,
        david_execution=execution_output,
        actions=extracted_actions_output.replace('```','')
    )
    print(f'New Prompt: {meta_output}')
    constraints, tips = get_new_instructions(meta_output)
    print(f'New Constraints: {constraints}')
    print(f'New Tips: {tips}')