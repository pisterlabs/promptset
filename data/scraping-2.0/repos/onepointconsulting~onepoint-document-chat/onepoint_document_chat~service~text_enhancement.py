from typing import List
from pathlib import Path
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.schema import Document

from onepoint_document_chat.config import cfg
from onepoint_document_chat.log_init import logger
from onepoint_document_chat.toml_support import prompts_toml


section = prompts_toml["text_enhancement"]


class NormalizedText(BaseModel):
    normalized_text: str = Field(
        description="Normalized text without line breaks and typos or unexpected punctuation."
    )


def prompt_factory_text_enhancements() -> ChatPromptTemplate:
    human_message = section["human_message"]
    prompt_msgs = [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=section["system_message"], input_variables=[]
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=human_message,
                input_variables=["text"],
            )
        ),
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


def chain_factory_initial_question() -> LLMChain:
    return create_structured_output_chain(
        NormalizedText,
        cfg.llm,
        prompt_factory_text_enhancements(),
        verbose=cfg.verbose_llm,
    )


enhance_text_chain = chain_factory_initial_question()


def enhance_text(file: Path) -> Path:
    text = file.read_text("utf-8")
    res: NormalizedText = enhance_text_chain.run({"text": text})
    normalized_text = res.normalized_text
    new_file = cfg.enhanced_text_folder / f"{file.stem}_enhanced.txt"
    new_file.write_text(normalized_text, encoding="utf-8")
    return new_file


def enhance_document(document: Document) -> Document:
    text = document.page_content
    try:
        res: NormalizedText = enhance_text_chain.run({"text": text})
        normalized_text = res.normalized_text
        logger.info(normalized_text)
        return Document(page_content=normalized_text, metadata=document.metadata)
    except Exception as e:
        logger.exception(f"Failed to normalize text from {document}")
        return document


def enhance_documents(documents: List[Document]) -> List[Document]:
    return [enhance_document(d) for d in documents]


if __name__ == "__main__":
    sample_file = Path(
        "data/example_extracts/Onepoint - Client story - Reaching for the skies (1)_2.txt"
    )

    enhanced_file = enhance_text(sample_file)
    logger.info(enhanced_file.read_text())
