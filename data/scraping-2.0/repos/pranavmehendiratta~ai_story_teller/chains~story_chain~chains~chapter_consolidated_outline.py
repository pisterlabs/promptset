from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.chapter_consolidated_outline.chapter_consolidated_outline_system_prompt import system_prompt
from chains.story_chain.prompts.chapter_consolidated_outline.chapter_consolidated_outline_human_prompt import human_prompt
from langchain.chains.sequential import SequentialChain
from typing import List, Dict, Any
from langchain.schema.document import Document
import json
from common.utils import concatenate_string_using_and

_messages = [
    SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm = ChatOpenAI(temperature = 0.5, model = "gpt-4")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_chapter_consolidated_outline_chain = LLMChain(
    llm = _llm,
    prompt = _chat_prompt,
    verbose = True
)

def chapter_consolidated_outline_chain_v1(
    topic: str,
    logical_sequence_dict: Dict[str, Any],
    formatted_outline: str
) -> Dict[str, Any]:
    logical_order = concatenate_string_using_and(logical_sequence_dict["logical_sequence"])
    consolidated_outline_str = _chapter_consolidated_outline_chain.run(
        {
            "topic": topic,
            "logical_order": logical_order,
            "logical_order_reason": logical_sequence_dict["explanation"],
            "formatted_outline": formatted_outline
        }
    )
    print(f"---------- Chapter wise consolidated outline----------")
    print(f"{consolidated_outline_str}")
    print("---------- End Chapter wise Outline ----------")
    return json.loads(consolidated_outline_str)