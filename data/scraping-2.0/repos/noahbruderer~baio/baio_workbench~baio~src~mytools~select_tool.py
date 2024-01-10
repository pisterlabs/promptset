import urllib.parse
from typing import Optional
from typing import Optional
from pydantic import BaseModel, Field
from typing import Optional
from src.non_llm_tools.utilities import log_question_uuid_json
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from pydantic import BaseModel, Field
from typing import Callable, Optional
from src.llm import LLM

llm = LLM.get_instance()

class MyTool(BaseModel):
    name: str = Field(
        default=""
    )
    func: Optional[Callable[..., str]] = Field(
        default=None, 
    )
    description: str = Field(
        default = "", 
        description="The description of the tool"
    )

class ToolSelector(BaseModel):
    name: str = Field(
        default="",
        description="The name of the best fitting tool to answer the question"
    )
    description: str = Field(
        default = "", 
        description="The description of the best fitting tool tool"
    )
    
    
def select_best_fitting_tool(question: str, tools: list):
    """FUNCTION to select tool to answer user questions"""
    BLAST_structured_output_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    runnable = create_structured_output_runnable(ToolSelector, llm, BLAST_structured_output_prompt)
    #retrieve relevant info to question
    #keep top 3 hits
    selected_tool = runnable.invoke({"input": f"{question} based on {tools}"})
    return selected_tool
