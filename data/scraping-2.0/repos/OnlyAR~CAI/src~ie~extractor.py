# @File Name:     extractor
# @Author :       Jun
# @date:          2023/10/27
# @Description :
import loguru
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from config import config
from ie.template import *
from utils.openai_utils import openai_llm, encoding

logger = loguru.logger


def extract_description(text, info, args):
    roles = info['roles']
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=config['max_token'],
        chunk_overlap=128,
        length_function=lambda x: len(encoding(x))
    )
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]

    descriptions = {}
    for role in roles:
        temp_info = {k: v for k, v in info.items() if isinstance(v, str)}
        temp_info['role'] = role
        temp_info['input_documents'] = docs
        if args.summary == 'mapreduce':
            map_prompt = PromptTemplate.from_template(DESCRIPTION_TEMPLATE)
            combine_prompt = PromptTemplate.from_template(DESCRIPTION_SUMMARY_TEMPLATE)
            map_reduce_chain = load_summarize_chain(
                openai_llm(),
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                return_intermediate_steps=True,
                verbose=args.debug
            )
            descriptions[role] = map_reduce_chain(temp_info, return_only_outputs=True)['output_text']
        elif args.summary == 'refine':
            refine_prompt = PromptTemplate.from_template(REFINE_DESCRIPTION_TEMPLATE)
            question_prompt = PromptTemplate.from_template(DESCRIPTION_TEMPLATE)
            refine_chain = load_summarize_chain(
                openai_llm(),
                chain_type="refine",
                question_prompt=question_prompt,
                refine_prompt=refine_prompt,
                return_intermediate_steps=True,
                verbose=args.debug
            )
            descriptions[role] = refine_chain(temp_info, return_only_outputs=True)['output_text']
        else:
            raise ValueError(f'No such summary type: {args.summary}')
    return descriptions


def extract_dialogue(text, info, args):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=config['max_token'],
        chunk_overlap=128,
        length_function=lambda x: len(encoding(x))
    )
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    context_prompt = PromptTemplate.from_template(DIALOGUE_SUMMARY_TEMPLATE)
    dialog_prompt = PromptTemplate.from_template(DIALOGUE_TEMPLATE)

    dialogues = []
    for doc in docs:
        context_chain = LLMChain(llm=openai_llm(), prompt=context_prompt)
        dialog_chain = LLMChain(llm=openai_llm(), prompt=dialog_prompt)
        temp_info = {k: v for k, v in info.items() if isinstance(v, str)}
        temp_info['text'] = doc.page_content
        dialogues.append({
            "context": context_chain(temp_info, return_only_outputs=True)['text'],
            "dialogue": dialog_chain(temp_info, return_only_outputs=True)['text']
        })
    return dialogues
