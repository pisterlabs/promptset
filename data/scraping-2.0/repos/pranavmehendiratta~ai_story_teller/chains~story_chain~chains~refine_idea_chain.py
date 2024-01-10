from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.refine_idea.refine_idea_human_prompt import human_prompt
from chains.story_chain.prompts.refine_idea.refine_idea_system_prompt import system_prompt
from typing import List, Dict
import json

_messages = [
    SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm = ChatOpenAI(temperature = 0.7, model = "gpt-4")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_refined_idea_chain = LLMChain(
    llm = _llm,
    prompt = _chat_prompt,
    verbose = True
)

def refine_idea_chain_v1(
    topic: str
) -> Dict[str, List[str]]:
    refined_ideas_str = _refined_idea_chain.run(
        {
            "topic": topic
        }
    )
    print("------- Refined Ideas -------")
    print(refined_ideas_str)
    print("------- Done with Refined Ideas -------")
    refined_ideas_json = json.loads(refined_ideas_str)
    return refined_ideas_json
