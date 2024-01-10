import os
from functools import partial
from typing import Any

from dotenv import load_dotenv
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import Runnable

load_dotenv()

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-tracing2"


doc_prompt = PromptTemplate.from_template("{page_content}")
partial_format_document = partial(format_document, prompt=doc_prompt)

# Assumes "stuffing" happens in pre-process step, and chain runs on one big document
stuff_chain: Runnable[Any, str]  = (
    {"content": partial_format_document}
    | PromptTemplate.from_template("Summarize the following content:\n\n{content}")
    | ChatOpenAI(model='gpt-3.5-turbo-1106').with_config({'callbacks': [ConsoleCallbackHandler()]})
    | StrOutputParser()
).with_config(run_name="Summarize (return doc)")