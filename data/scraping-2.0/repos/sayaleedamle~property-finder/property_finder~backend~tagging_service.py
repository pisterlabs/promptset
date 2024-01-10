from langchain.chains import LLMChain
from langchain.chains import create_tagging_chain_pydantic
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from property_finder.configuration.config import cfg
from property_finder.configuration.log_factory import logger
from property_finder.backend.model import ResponseTags
from property_finder.configuration.toml_support import read_prompts_toml

prompts = read_prompts_toml()


def prompt_factory_sentiment() -> ChatPromptTemplate:
    section = prompts["tagging"]
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
                input_variables=["answer"],
            )
        ),
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


def sentiment_chain_factory() -> LLMChain:
    return create_tagging_chain_pydantic(
        ResponseTags, cfg.llm, prompt_factory_sentiment(), verbose=True
    )


chain = create_tagging_chain_pydantic(ResponseTags, cfg.llm, prompt_factory_sentiment())


def prepare_sentiment_input(question: str) -> dict:
    return {"text": question}


def tag_response(response: str):
    res = chain(prepare_sentiment_input(response))
    return res


if __name__ == "__main__":

    def process_answer(answer: str):
        logger.info(type(answer))
        logger.info(answer)

    # Does your organization support an event driven architecture for data integration?

    response_tags: ResponseTags = sentiment_chain_factory().run(
        prepare_sentiment_input("yes")
    )
    process_answer(tag_response(response_tags.is_positive)["answer"])
