from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.chapters.chapters_system_prompt import system_prompt
from chains.story_chain.prompts.chapters.chapters_human_prompt import human_prompt
from typing import List
import json

_messages = [
    SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_chapters_chain = LLMChain(
    llm = _llm,
    prompt = _chat_prompt,
    verbose = True
)

def chapters_chain_v1(
    topic: str,
    section_number: str,
    chapter_name: str,
    chapter_number: str,
    section_name: str,
    ideas: str,
    previous_section_title: str,
    previous_section_content: str,
    next_section_title: str,
    content: str
) -> str:
    section_summary_str = _chapters_chain.run(
        {
            "topic": topic,
            "section_number": section_number,
            "chapter_name": chapter_name,
            "chapter_number": chapter_number,
            "section_name": section_name,
            "ideas": ideas,
            "previous_section_title": previous_section_title,
            "previous_section_content": previous_section_content,
            "next_section_title": next_section_title,
            "content": content
        }
    )
    print(f"---------- Chapter {chapter_name} Section {section_name} Summary ----------")
    print(section_summary_str)
    print("---------- End of Section Summary ----------")
    return section_summary_str