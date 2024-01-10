from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.logical_sequence.logical_sequence_human_prompt import human_prompt
from chains.story_chain.prompts.logical_sequence.logical_sequence_system_prompt import system_prompt
from typing import List, Dict, Any
from langchain.schema.document import Document
import json

_messages = [
    SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm = ChatOpenAI(temperature = 0, model = "gpt-4")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_logical_sequence_chain = LLMChain(
    llm = _llm,
    prompt = _chat_prompt,
    verbose = True
)

def logical_sequence_chain_v1(
    topic: str
) -> Dict[str, Any]:
    logical_sequence_str = _logical_sequence_chain.run(
        {
            "topic": topic    
        }
    )
    print("------- Logical Sequence -------")
    print(logical_sequence_str)
    print("------- Done with Logical Sequence -------")
    logical_sequence_json = json.loads(logical_sequence_str)
    return logical_sequence_json
