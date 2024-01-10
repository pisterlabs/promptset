from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

import logging


from backends.utils._openai import OPENAI_CLIENT_OPTS, OPENAI_PROMPT_OPTS
from backends.utils.exceptions import (
    SINGLE_PROMPT_REFINE_SUMMARIZATION_FAILED,
)
from backends.utils.create_document import create_document

logger = logging.getLogger("refinement")


def single_prompt(text: str, model: str, section_strings: str) -> str:
    try:
        text_len = len(text)

        logger.info(
            f"Beginning single-prompt refinement of summarization length {text_len} using the {model} backend"
        )

        llm = ChatOpenAI(
            **OPENAI_CLIENT_OPTS,
            **OPENAI_PROMPT_OPTS,
            model_name=model,
        )

        text = create_document(text, text_len, stuff=True)

        refine_summary_template = (
            "Your job is to reformat an exhaustive summary into 3 concise sections. "
            "Separate each section with a newline character: "
            + section_strings
            + "The summary should preserve all information and context, numbers, dates, locations and names. "
            "The following is the summary to be reformatted: {text}"
        )
        refine_summary_prompt = PromptTemplate.from_template(refine_summary_template)

        llm_chain = LLMChain(llm=llm, prompt=refine_summary_prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        summary = stuff_chain.run(text).strip()

        logger.info(
            f"Completed single-prompt refinement of summary length {text_len} using the {model} backend"
        )

        return summary

    except Exception as e:
        logger.error(f"{SINGLE_PROMPT_REFINE_SUMMARIZATION_FAILED.detail}: {e}")
        raise SINGLE_PROMPT_REFINE_SUMMARIZATION_FAILED
