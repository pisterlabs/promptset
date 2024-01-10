"""
    Refine summary
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0903,W1203

import time
from dataclasses import dataclass
import tiktoken
import traceback
import logging

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from utils.utils import parse_llm_xml

logger : logging.Logger = logging.getLogger()

@dataclass
class RefineResult:
    """Result of refine"""
    summary     : str
    tokens_used : int
    error       : str

refine_initial_prompt_template = """\
You are the company's press secretary.
Write a concise summary of the text (delimited with XML tags).
Use only provided text, do not add anythong from yourself.
Try to extract as much as possible useful information from provided text.
Make sure a summary is entirely in English. Translate it into English where needed.
Rewrite it if a summary exceeds 2000 characters. Do not repeat the same information.
If the text does not contain any information to summarize say "No summary".

<input_text>
{input_text}
</input_text>

Please provide result in XML format:
<summary>
Summary here (not more than 2000 characters)
</summary>
"""

refine_combine_prompt_template = """\
You are the company's press secretary.
Your job is to produce a final summary from existed summary (delimited with XML tags) and some new context (delimited with XML tags).
If new conext is not useful, just say that it's not useful.
Make sure a summary is entirely in English. Translate it into English where needed.
Rewrite it if a summary exceeds 2000 characters. Do not repeat the same information.

Please provide result in XML format:
<not_useful>
    True if new context was not useful, False if new content was used
</not_useful>
<refined_summary>
    Refined summary here ONLY if new context was useful (not more than 2000 characters)
</refined_summary>

<existing_summary>
{existing_summary}
</existing_summary>

<more_context>
{more_context}
</more_context>
"""

@dataclass
class RefineInitialResult:
    """Result of initial refine"""
    summary     : str
    tokens_used : int

@dataclass
class RefineStepResult:
    """Result of refine step"""
    new_summary : str
    tokens_used : int
    useful      : bool

class RefineChain():
    """Refine chain"""
    refine_initial_chain : LLMChain
    refine_combine_chain : LLMChain
    token_estimator : tiktoken.core.Encoding
    TOKEN_BUFFER = 50

    def __init__(self, llm : ChatOpenAI):
        if llm:
            refine_initial_prompt = PromptTemplate(template= refine_initial_prompt_template, input_variables=["input_text"])
            self.refine_initial_chain = LLMChain(llm= llm, prompt= refine_initial_prompt)
            
            refine_combine_prompt = PromptTemplate(template= refine_combine_prompt_template, input_variables=["existing_summary", "more_context"])
            self.refine_combine_chain = LLMChain(llm= llm, prompt= refine_combine_prompt)

            self.token_estimator = tiktoken.encoding_for_model(llm.model_name)

    def len_function(self, text : str) -> int:
        """Lenght function"""
        return len(self.token_estimator.encode(text))

    def get_max_possible_index(self, sentence_list : list[str], start_index : int, max_tokens : int, len_function : any) -> int:
        """Find next possible part of text"""
        token_count = 0
        for sentence_index in range(start_index, len(sentence_list)):
            token_count_p = len_function(sentence_list[sentence_index])
            token_count = token_count + token_count_p
            if token_count <= max_tokens:
                continue
            return sentence_index
        return len(sentence_list)
    
    def refine(self, text : str, max_tokens : int) -> RefineResult:
        """Refine call"""
        sentence_list = text.replace('\n', '.').split('.')
        sentence_list_without_dupl = []
        for sentence in sentence_list:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence_list_without_dupl) == 0 or sentence_list_without_dupl[-1] != sentence:
                sentence_list_without_dupl.append(sentence)
        sentence_list = [f'{s}. ' for s in sentence_list_without_dupl]

        tokens_used = 0
        summary = ""

        try:
            current_index = 0
            summary_step  = True
            for _ in range(len(sentence_list)+1):

                # execute first step -  summary
                if summary_step:
                    prompt_len = self.len_function(self.refine_initial_chain.prompt.format(input_text = ''))
                    max_token_in_text = max_tokens - prompt_len - self.TOKEN_BUFFER
                    new_index = self.get_max_possible_index(
                        sentence_list, 
                        current_index,
                        max_token_in_text,
                        self.len_function
                    )
                    status = f'--- Process doc init {current_index}:{new_index} / {len(sentence_list)}'
                    logger.info(status)

                    current_doc_list = sentence_list[current_index:new_index]
                    current_doc = ''.join(current_doc_list)
                    current_doc_len = self.len_function(current_doc)

                    logger.debug(f'max_tokens={max_tokens}, prompt_len={prompt_len}, max_token_in_text={max_token_in_text}, current_doc_len={current_doc_len}')
                    refine_initial_result = self.execute_initial_refine(current_doc)
                    tokens_used += refine_initial_result.tokens_used
                    summary = refine_initial_result.summary

                    time.sleep(0.1)

                    current_index = new_index+1
                    if new_index >= len(sentence_list):
                        break

                    if not summary: # wait for valuable summary first
                        continue

                    summary_step = False
                    continue

                # execute refine
                prompt_len = self.len_function(self.refine_combine_chain.prompt.format(existing_summary = summary, more_context = ''))
                max_token_in_text = max_tokens - prompt_len - self.TOKEN_BUFFER
                new_index = self.get_max_possible_index(
                    sentence_list, 
                    current_index, 
                    max_token_in_text,
                    self.len_function
                )
                status = f'--- Process doc refine {current_index}:{new_index} / {len(sentence_list)}'
                logger.info(status)

                current_doc = ''.join(sentence_list[current_index:new_index])

                refine_step_result = self.execute_refine_step(summary, current_doc)
                tokens_used += refine_step_result.tokens_used
                if refine_step_result.useful:
                    if refine_step_result.new_summary:
                        summary = refine_step_result.new_summary
                    else:
                        logger.error('ERROR: empty summary with Useful flag')

                time.sleep(0.1)

                current_index = new_index+1
                if new_index >= len(sentence_list):
                    break
            
            return RefineResult(summary, tokens_used, None)
        except Exception as error: # pylint: disable=W0718
            logger.error(f'Error: {error}. Track: {traceback.format_exc()}')
            return RefineResult(summary, tokens_used, error)

    def execute_initial_refine(self, document : str) -> RefineInitialResult:
        """Execute refine step"""
        tokens_used    = 0
        summary        = ''

        logger.info('------- execute_initial_refine')

        refine_initial_result  = None
        try:
            with get_openai_callback() as cb:
                refine_initial_result = self.refine_initial_chain.run(input_text = document)
            tokens_used = cb.total_tokens
            logger.debug(f'refine_initial_result={refine_initial_result}')
        except Exception as error: # pylint: disable=W0718
            logger.error(error)

        if refine_initial_result:
            summary_xml = parse_llm_xml(refine_initial_result, ["summary"])
            summary_str = summary_xml["summary"].strip()
            
            if "No summary" in summary_str:
                if len(summary_str) < 20: # fix hallucination - in some cases model adds this wording even in normal summary
                    return RefineInitialResult('', tokens_used)
                logger.warning('Wording <No summary> in real summary.')
                summary_str = summary_str.replace("No summary", '')
            summary = summary_str
        return RefineInitialResult(summary, tokens_used)

    def execute_refine_step(self, existing_summary : str, more_context : str) -> RefineStepResult:
        """Execute refine step"""
        tokens_used    = 0
        refined_useful = False
        summary        = ''

        logger.info('------- execute_refine_step')

        refine_step_result  = None
        try:
            with get_openai_callback() as cb:
                refine_step_result = self.refine_combine_chain.run(existing_summary = existing_summary, more_context = more_context)
            tokens_used = cb.total_tokens
            logger.debug(refine_step_result)
        except Exception as error: # pylint: disable=W0718
            logger.error(error)

        if refine_step_result:
            refined_xml = parse_llm_xml(refine_step_result, ["not_useful", "refined_summary"])
            refined_summary = refined_xml["refined_summary"]
            refined_useful  = refined_xml["not_useful"].lower().strip() != "true"
            if refined_useful:
                summary = refined_summary

        return RefineStepResult(summary, tokens_used, refined_useful)


