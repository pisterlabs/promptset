from bioelectronica.normdata.bacstr_prompts import _sys, _human
from bioelectronica._log import logger_normdata

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.callbacks import get_openai_callback

import os


chat_prompt = ChatPromptTemplate.from_messages(
  [
    SystemMessagePromptTemplate.from_template(_sys),
    HumanMessagePromptTemplate.from_template(_human)
  ]
)

# class OutputParser(BaseOutputParser):
#     def parse(self, text: str):
#         print(text)
#         import json
#         return json.loads(text.strip())

llm_chain = LLMChain(
    llm=ChatOpenAI(model_name=os.getenv('OPENAI_API_MODEL'), temperature=0),
    prompt=chat_prompt,
    # output_parser=OutputParser()
)

def norm_bacstr(_input):
    if len(_input) < 2000:
        with get_openai_callback() as cb:
            chat_prompt.format_messages(input=_input)
            _re = llm_chain(inputs={"input": _input}) # return_only_outputs=True
            _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        logger_normdata.info(_re)
        logger_normdata.info(_token_cost)
        return _re["text"]
    else:
        logger_normdata.error("input too long: {_input}")

