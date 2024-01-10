import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import ChatPromptTemplate

load_dotenv()

SUMMARY_PROMPT = """
    Summarize this text.
    
 CONVERSATION TEXT: 

 {text}

"""


def create_component_summary_chain():
    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    model = ChatOpenAI(temperature=0,
                       model_name="gpt-3.5-turbo-16k")
    chain = prompt | model
    return chain

GLOBAL_SUMMARY_PROMPT = """
    Summarize these summaries. 
    
LIST OF SUMMARIES TEXT: 
+++++ 
{text}
+++++

That was the SUMMARIES TEXT. As a reminder, we are writing an academic review article based on the SUMMARIES TEXT.
"""

def create_global_summary_chain():
    prompt = ChatPromptTemplate.from_template(GLOBAL_SUMMARY_PROMPT)

    model = ChatOpenAI(temperature=0,
                       model_name="gpt-3.5-turbo-16k")
    chain = prompt | model
    return chain

async def summarize_playbook(playbook_text_file_path: str):
    output_text = ""
    component_summaries_text = ""

    component_summary_chain = create_component_summary_chain()
    global_summary_chain = create_global_summary_chain()

    # Load playbook text file from path
    with open(playbook_text_file_path, "r") as file:
        playbook_text = file.read()

    chapters = playbook_text.split("CHAPTER")
    # The split will create an empty string at the start, so remove it
    if chapters[0] == '':
        chapters.pop(0)

    chapter_dict = {f"CHAPTER {i + 1}": chapter for i, chapter in enumerate(chapters)}

    # Process them using chain.abatch
    print("Summarizer is summarizing.")

    all_summaries = await component_summary_chain.abatch(inputs=[{"text": chapter_text} for chapter_text in chapter_dict.values()])
    for summary in all_summaries:
        print(f"chunk:\n\n{summary.content}\n\n===\n\n")
        component_summaries_text += summary.content + "\n\n"
    print("Big summarizer is big summarizing.")

    global_summary = await global_summary_chain.ainvoke(input={"text": component_summaries_text})
    print(f"{global_summary.content}")

    document_file_name = f"document_summary.md"
    file_number = 0
    while Path(document_file_name).is_file():
        document_file_name = f"document_summary{file_number}.md"
        file_number += 1

    with open(document_file_name, "w", encoding="utf-8") as file:
        file.write(global_summary.content)

if __name__ == "__main__":
    playbook_text_file_in = "/Users/paul/Documents/Research/Palestine/zionist-playbook-cloudconvert.txt"

    asyncio.run(summarize_playbook(playbook_text_file_path=playbook_text_file_in))
