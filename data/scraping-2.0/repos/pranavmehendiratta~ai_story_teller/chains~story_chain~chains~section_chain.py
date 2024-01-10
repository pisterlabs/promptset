from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.section.section_human_prompt import human_prompt_gpt_4, human_prompt
from chains.story_chain.prompts.section.section_system_prompt import system_prompt_gpt_4, system_prompt
from typing import List
import json

_messages_gpt_4 = [
    SystemMessagePromptTemplate(prompt = system_prompt_gpt_4), 
    HumanMessagePromptTemplate(prompt = human_prompt_gpt_4)
]
_llm_gpt_4 = ChatOpenAI(temperature = 0.7, model = "gpt-4")
_chat_prompt_gpt_4 = ChatPromptTemplate.from_messages(messages = _messages_gpt_4)
_section_chain_gpt_4 = LLMChain(
    llm = _llm_gpt_4,
    prompt = _chat_prompt_gpt_4,
    verbose = True
)

_messages_gpt_3_5 = [
    SystemMessagePromptTemplate(prompt = system_prompt), 
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm_gpt_3_5 = ChatOpenAI(temperature = 0.7, model = "gpt-3.5-turbo-16k")
_chat_prompt_gpt_3_5 = ChatPromptTemplate.from_messages(messages = _messages_gpt_3_5)
_section_chain_gpt_3_5 = LLMChain(
    llm = _llm_gpt_3_5,
    prompt = _chat_prompt_gpt_3_5,
    verbose = True
)

def section_chain_v1(
    topic: str,
    context: str,
    structure: str,
    style: str,
    narration: str,
    tone: str,
    initial_opening: str,
    section_description: List[str],
    section_topic_dir_path: str,
    use_gpt_4: bool
) -> None:
    current_section_opening = initial_opening
    for index, section in enumerate(section_description):
        print(f"Working on section {index}")
        section_file_name = f"{section_topic_dir_path}/{index}.json"
        topics_to_cover_in_next_section = section_description[index + 1] if index + 1 < len(section_description) else "This is the last section. End the podcast here."
        section_content_str = _style_chain_section_helper_v1(
            topic = topic,
            context = context,
            structure = structure,
            style = style,
            narration = narration,
            tone = tone,
            current_section_opening = current_section_opening,
            topics_to_cover_in_current_section = section,
            topics_to_cover_in_next_section = topics_to_cover_in_next_section,
            use_gpt_4 = use_gpt_4
        )
        print(f"__________ Section {index} __________")
        print(section_content_str)
        print("____________________________________")

        with open(section_file_name, "w") as f:
            f.write(section_content_str)

        print("section_content_str written to file")
        section_content_json = json.loads(section_content_str)
        print("current_section_opening converted to json")
        next_section_content = str(section_content_json["next_section_opening"])
        print("next_section_opening extracted from json")
        current_section_opening = next_section_content


def _style_chain_section_helper_v1(
    topic: str,
    context: str,
    structure: str,
    tone: str,
    style: str,
    narration: str,
    current_section_opening: str,
    topics_to_cover_in_current_section: str,
    topics_to_cover_in_next_section: str,
    use_gpt_4: bool
):
    if use_gpt_4:
        return _section_chain_gpt_4.run(
            {
                "topic": topic,
                "structure": structure,
                "style": style,
                "narration": narration,
                "tone": tone,
                "current_section_opening": current_section_opening,
                "topics_to_cover_in_current_section": topics_to_cover_in_current_section,
                "topics_to_cover_in_next_section": topics_to_cover_in_next_section
            }
        )
    else:
        return _section_chain_gpt_3_5.run(
            {
                "topic": topic,
                "context": context,
                "structure": structure,
                "style": style,
                "narration": narration,
                "tone": tone,
                "current_section_opening": current_section_opening,
                "topics_to_cover_in_current_section": topics_to_cover_in_current_section,
                "topics_to_cover_in_next_section": topics_to_cover_in_next_section
            }
        )