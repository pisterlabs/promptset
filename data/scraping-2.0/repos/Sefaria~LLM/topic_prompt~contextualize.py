"""
Provide context for a source
"""
import django
django.setup()
from sefaria.model import *
from util.general import get_ref_text_with_fallback, get_by_xml_tag
import re

import langchain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def context_from_section(segment_oref: Ref) -> str:
    """
    Given a segment ref, provide context from its section ref
    :param segment_oref:
    :return: context from section ref
    """
    segment_text = get_ref_text_with_fallback(segment_oref, "en", auto_translate=True)
    section_text = get_ref_text_with_fallback(segment_oref.section_ref(), "en", auto_translate=True)
    system_message = SystemMessage(content="# Identity\nYou are a Jewish scholar familiar with all Torah texts.\n"
                                           "# Task\nGiven a segment of Torah text and surrounding context text, output"
                                           "a summary of the relevant parts of the context that will help users"
                                           " understand the segment of Torah text."
                                           "# Input format\n"
                                           "- Segment of Torah text surrounded by <segment> XML tags.\n"
                                           "- Context text surrounded by <context> XML tags.\n"
                                           "# Output format\n"
                                           "Summary of the relevant context text to help users understand <segment>"
                                           " text. Output should be surrounded in <relevant_context> XML"
                                           " tags. No more than 50 words. Summary should start with the word 'The"
                                           " context describes'.")
    human_message = HumanMessage(content=f"<segment>{segment_text}</segment>\n"
                                         f"<context>{section_text}</context>")
    llm = ChatAnthropic(model="claude-2", temperature=0, max_tokens_to_sample=100000)
    response = llm([system_message, human_message])
    context = get_by_xml_tag(response.content, "relevant_context")
    if context is None:
        return response.content
    context = re.sub(r"^The context describes ", "", context)
    return context


def context_from_liturgy(oref):
    text = get_ref_text_with_fallback(oref, "en", auto_translate=True)
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_message = SystemMessage(content="""
    Given a text from the Jewish cannon, add any relevant context that would help a user understand this text from a
    Jewish perspective. Relevant context may be:
    If this text is a prayer, when was it recited and why?
    Historical significance to Jewish history
    How this text is viewed nowadays by Jewish people
    
    DO NOT offer an interpretation or explanation of the text. Only offer helpful context.
    
    Limit to 50 words or less.
    """)

    prompt = PromptTemplate.from_template("Citation: {citation}\nText: {text}")
    human_message = HumanMessage(content=prompt.format(text=text, citation=oref.normal()))
    response = llm([system_message, human_message])
    return response.content


def get_context(oref: Ref):
    if oref.primary_category == "Tanakh":
        context = context_from_section(oref)
    else:
        context = context_from_liturgy(oref)
    return context


if __name__ == '__main__':
    print(get_context(Ref("Nehemiah 8:14-16")))
    # print(context_from_section(Ref("Nehemiah 8:14-16")))

