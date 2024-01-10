from pathlib import Path
from langchain.chains import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from knowledge_graph.configuration.toml_support import read_prompts_toml
from knowledge_graph.configuration.config import cfg
import knowledge_graph.backend.qna_service as qna_service
import knowledge_graph.services.vector_db as v_db

prompts = read_prompts_toml()


def prompt_factory() -> ChatPromptTemplate:
    """Prompt Factory for finding answer to questions"""
    section = prompts["qna_bot"]
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
                input_variables=["content", "summary", "question"],
            )
        ),
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


async def return_answer(query: str):
    """Chain function for finding answer to question"""

    #db_text, db_summary = v_db.create_embeddings_text(input_text, summary_path)
    content_text, content_summary = await v_db.similarity_search(query)
    prompt = prompt_factory()
    chain = LLMChain(llm=cfg.llm, prompt=prompt)
    return await chain.arun({"content": content_text, "summary": content_summary, "question": query})
