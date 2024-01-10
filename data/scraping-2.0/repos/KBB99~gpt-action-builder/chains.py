from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from prompts import get_evaluation_prompt_template, get_extracted_actions_prompt_template

def initialize_chain(prompt_template):
    """Initializes and returns a goal evaluation chain."""
    evaluation_prompt = PromptTemplate(
        input_variables=["execution_output", "goal"], 
        template=prompt_template
    )
    evaluation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"), 
        prompt=evaluation_prompt, 
        verbose=True, 
    )
    return evaluation_chain

def initialize_chains():
    evaluation_prompt_template = get_evaluation_prompt_template()
    extracted_actions_prompt = get_extracted_actions_prompt_template()
    return initialize_chain(evaluation_prompt_template), initialize_chain(extracted_actions_prompt)
