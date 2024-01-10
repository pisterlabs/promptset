from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.refined_summary.refined_summary_system_prompt import system_prompt
from chains.story_chain.prompts.refined_summary.refined_summary_human_prompt import human_prompt
from langchain.chains.sequential import SequentialChain
from typing import List, Dict, Any
from langchain.schema.document import Document
import json
from common.utils import concatenate_string_using_and

_messages = [
    #SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm = ChatOpenAI(temperature = 0, model = "gpt-4")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_refined_summary_chain = LLMChain(
    llm = _llm,
    prompt = _chat_prompt,
    verbose = True
)

def refined_summary_chain_v1(
    topic: str,
    logical_sequence_dict: Dict[str, Any],
    summary: str,
) -> Dict[str, Any]:
    logical_order = concatenate_string_using_and(logical_sequence_dict["logical_sequence"])
    refined_summary = _refined_summary_chain.run(
        {
            "topic": topic,
            "logical_order": logical_order,
            "logical_order_reason": logical_sequence_dict["explanation"],
            "summary": summary
        }
    )
    print(f"---------- Chapter wise refined summary ----------")
    print(f"{refined_summary}")
    print("---------- End Chapter refined summary ----------")
    return json.loads(refined_summary)