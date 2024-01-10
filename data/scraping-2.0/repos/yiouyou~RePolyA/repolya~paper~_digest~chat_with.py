from repolya._const import PAPER_PROMPT

from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import load_prompt

import os


def chat_with_openai_4(_sys, _human, _info):
    sys_file = str(PAPER_PROMPT / _sys)
    human_file = str(PAPER_PROMPT / _human)
    sys_prompt = SystemMessagePromptTemplate.from_template_file(
        sys_file,
        input_variables=[]
    )
    human_prompt = HumanMessagePromptTemplate.from_template_file(
        human_file,
        input_variables=["info"]
    )
    _prompt = ChatPromptTemplate.from_messages(
        [sys_prompt, human_prompt]
    )
    with get_openai_callback() as cb:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        chain = LLMChain(llm=llm, prompt=_prompt)
        _res = chain.run(info=_info)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        _res_step = f"{_token_cost}\n\n" + "="*20+" prompt "+"="*20+"\n" + _prompt.format(info=_info) + "="*20+" prompt "+"="*20+"\n" + f"response:\n\n" + "\n".join(_res)
    return _res, _res_step
