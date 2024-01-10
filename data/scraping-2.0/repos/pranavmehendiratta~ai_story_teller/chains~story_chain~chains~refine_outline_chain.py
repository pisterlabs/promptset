from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from chains.story_chain.prompts.refine_outline.refine_outline_human_prompt import human_prompt
from chains.story_chain.prompts.refine_outline.refine_outline_system_prompt import system_prompt
from typing import List


OUTLINE_SEPARATOR = "\n\n_____ NEXT OUTLINE _____\n\n"
OUTLINE_CONTEXT = "outlines"

_messages = [
    SystemMessagePromptTemplate(prompt = system_prompt),
    HumanMessagePromptTemplate(prompt = human_prompt)
]
_llm_16k = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")
_chat_prompt = ChatPromptTemplate.from_messages(messages = _messages)
_refine_outline_chain_16k = LLMChain(
    llm = _llm_16k,
    prompt = _chat_prompt,
    verbose = True
)

def refine_outline_chain_v1(
    outlines: List[str]
) -> str:
    joined_outlines = OUTLINE_SEPARATOR.join(outlines)
    refined_outline = _refine_outline_chain_16k.run(
        {
            "outline_separator": OUTLINE_SEPARATOR,
            "outlines": joined_outlines
        }
    )
    
    print("---------- Refined Outline ----------")
    print(refined_outline)
    print("---------- End Refined Outline ----------")

    return refined_outline