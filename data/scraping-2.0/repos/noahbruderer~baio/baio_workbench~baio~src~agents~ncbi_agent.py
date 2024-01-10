from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from src.mytools.select_tool import select_best_fitting_tool, MyTool
from src.mytools.eutils_tool import eutils_tool
from src.mytools.BLAST_structured_output import blast_tool
from src.mytools.BLAT_structured_output import BLAT_tool
from src.llm import LLM
import os
from langchain.chat_models import ChatOpenAI

llm = LLM.get_instance()
# ###Agent prompt
# prefix = """Have a conversation with a scientist, answering the following questions as best you can.
# ALWAYS PASS THE FULL USER QUESTION INTO THE TOOLs
# YOU ARE A WORLD CLASS MOLECULAR BIOLOGIST; TAXONOMIST; BIOINFORMATICIAN.
# You are executing api calls to get information from NCBI.
# The tools can only work if you input it the total input Question from below, do not condense it or reduce it
# """

# suffix = """Begin!"

# {chat_history}
# Question: {input}
# {agent_scratchpad}"""

# tools = [
#         Tool(
#         name="Eutils",
#         func=eutils_tool,
#         description="Always use this tool when you are making requests on NCBI except when you are given a DNA or protein sequence",
#     ),
#         Tool(
#         name="BLAST",
#         func=blast_tool,
#         description="With this tool you have access to the BLAST data base on NCBI, use it for any kind of query involving a DNA sequence",
#     ),
#         # Always use this tool when you are asked to find to which organism a specific sequence belongs to. If you cant find the answer using this tool, use the blast tool
#     ]


# prompt = ZeroShotAgent.create_prompt(
#     tools,
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["input", "chat_history", "agent_scratchpad"],
# )

# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# llm_chain = LLMChain(llm=llm, prompt=prompt)

# ncbi_agent = initialize_agent(tools, 
#                               llm, 
#                               agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
#                               verbose=True, 
#                               memory=memory)

tools = [
        MyTool(
        name="eutils_tool",
        func=eutils_tool,
        description="Always use this tool when you are making requests on NCBI except when you are given a DNA or protein sequence",
    ),
        MyTool(
        name="blast_tool",
        func=blast_tool,
        description="With this tool you have access to the BLAST data base on NCBI, use it for queries about a DNA or protein sequence\
        EXCEPT if the question is about aligning a sequence with a specifice organisms, then use BLAT_tool.",
    ),
    MyTool(
        name="BLAT_tool",
        func=BLAT_tool,
        description="Use for questions such s 'Align the DNA sequence to the human:ATTCGCC...; With this tool you have access to the ucsc genome data base. It can find where DNA sequences are aligned on the organisms genome, exact positions etc. ",
),
    ]

function_mapping = {
    "eutils_tool": eutils_tool,
    "blast_tool": blast_tool,
    "BLAT_tool": BLAT_tool
}

def ncbi_agent(question: str):
    print('In NCBI agent...\nSelecting tool...')
    selected_tool = select_best_fitting_tool(question, tools)
    function_to_call = function_mapping.get(selected_tool.name)
    print(f'Selected tool: {selected_tool.name}')
    answer = function_to_call(question)
    print(answer)
    return answer
