from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from config import TEMPERATURE, MODEL_NAME, TIMEOUT, STREAMING
from tools import tools
from callback_manager import CallbackManager
from custom_classes import CustomPromptTemplate, CustomOutputParser

# Initialize parser and callback manager
output_parser = CustomOutputParser()
cb = CallbackManager()

# Initialize language learning model
agent_llm = ChatOpenAI(
    temperature=TEMPERATURE, 
    model_name=MODEL_NAME,
    request_timeout=TIMEOUT,
    streaming=STREAMING,
)

def initialize_agent(david_instantiation_prompt: str):
    """Initializes agent with provided prompt.

    Args:
        david_instantiation_prompt: The prompt for initializing the agent.

    Returns:
        Agent executor object.
    """
    prompt = CustomPromptTemplate(
        template=david_instantiation_prompt,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "constraints", "tips", "intermediate_steps"]
    )
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=agent_llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor

def run_agent(goal, david_instantiation_prompt, constraints, tips):
    cb.clear()
    agent = initialize_agent(david_instantiation_prompt)
    try:
        agent.run(input=goal, constraints=constraints, tips=tips, callbacks=[cb])
    except Exception as e:
        print(f'Exception: {e}')
        print('Continuing...')
    return ''.join(cb.last_execution)