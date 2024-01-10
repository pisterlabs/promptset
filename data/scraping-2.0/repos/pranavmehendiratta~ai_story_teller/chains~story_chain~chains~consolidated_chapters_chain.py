from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.consolidated_chapters.consolidated_chapters_system_prompt import system_prompt
from chains.story_chain.prompts.consolidated_chapters.consolidated_chapters_human_prompt import human_prompt
import json
from typing import List

_messages = [
    SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm = ChatOpenAI(temperature = 0.3, model = "gpt-4")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_consolidated_chapters_chain = LLMChain(
    llm = _llm,
    prompt = _chat_prompt,
    verbose = True
)

def consolidated_chapters_summary_v1(
    topic: str,
    section_name: str,
    ideas: str,
    previous_section_name: str,
    next_section_name: str,
    summaries: str
) -> str:
    
    """
    prompt = _chat_prompt.format(
        topic = topic,
        section_name = section_name,
        ideas = ideas,
        previous_section_name = previous_section_name,
        next_section_name = next_section_name,
        summaries = summaries
    )
    print(f"---------- Consolidated '{section_name}' Summary ----------")
    print(prompt)
    print("---------- End of Consolidated Summary ----------")
    return ""
    """

    consolidated_chapter_summary = _consolidated_chapters_chain.run(
        {
            "topic": topic,
            "section_name": section_name,
            "ideas": ideas,
            "previous_section_name": previous_section_name,
            "next_section_name": next_section_name,
            "summaries": summaries
        }
    )
    print(f"---------- Consolidated '{section_name}' Summary ----------")
    print(consolidated_chapter_summary)
    print("---------- End of Consolidated Summary ----------")
    return consolidated_chapter_summary
