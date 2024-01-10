from langchain.chains import LLMChain
from gpu_llm import GpuLLM, LLMConfig

conf = LLMConfig(max_length=1000,
                do_sample=False,
                temperature=0.0,
                top_k=0,
                top_p=1,
                repetition_penalty=1.0,
                early_stopping=True,
                # _model = None,
                # _stop_regex = None,
                # stopping_criteria = "</s>",
                )


open_ai = GpuLLM(config=conf,
                    seed=153513,
                model_name = "WizardLM/WizardLM-70B-V1.0",
                device="auto",
                skip_validation=True,
                # stop_msgs=["</s>"],
                )

open_ai._initialize()

from langchain.agents import load_tools, initialize_agent,AgentType,Tool
from langchain.utilities import GoogleSearchAPIWrapper

GOOGLE_API_KEY =  "AIzaSyBSnAYoQ5w7oKE6s1F7j2MWiDcIJS7ZO0c"
GOOGLE_CSE_ID = "b759664f4a733494f"
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY,google_cse_id=GOOGLE_CSE_ID)
# TODO: add coding tools
tools = [
    Tool(
    name="Google",
    func=search.run,
    description="Use to perfrom Google search through out the internet"
    ),
    Tool(name="get_Income",
        func=lambda x : "user had earn 100$ by selling food",
        description="Use to get user income"
        ),
    Tool(name="get_Expense",
        func=lambda x : "user had spend 10$ by buying food",
        description="Use to get user expense"
        )
        
] + load_tools(
    ['llm-math'], llm=open_ai,
    )    

from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

agent_prefix = """You are a financial assistance. your job is to answer the user question.
You have access to the following tools:""" 
agent_suffix = """Begin!
{question_history}
Question: {input}
{agent_scratchpad}"""

tool_names = [tool.name for tool in tools]
agent_convhis = ConversationBufferWindowMemory(k=2,memory_key="question_history",output_key='output')
agent_prompt = ZeroShotAgent.create_prompt(tools=tools,
                                           prefix=agent_prefix, 
                                           suffix=agent_suffix,
                                           input_variables=["input",'agent_scratchpad','question_history'])
agent_custom = ZeroShotAgent(llm_chain=LLMChain(llm=open_ai,prompt=agent_prompt),allowed_tools=tool_names,return_intermediate_steps=True
                             ,max_iterations=1)
agent_commit = AgentExecutor.from_agent_and_tools(agent=agent_custom,tools=tools,
                                                verbose=True,
                                                  memory=agent_convhis,
                                                  # handle_parsing_errors=True,
                                                  return_intermediate_steps=True)


agent_commit({'input':"how much money dose the user have left in their bank account?"}) 
