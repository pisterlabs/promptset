# %%
from typing import Literal

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationTokenBufferMemory

from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from mtg.utils import get_openai_api_key

openai_api_key = get_openai_api_key()


DECKBUILDING_SYSTEM_MESSAGE = """
You are Nissa a Magic the Gathering Assistant that helps with deckbuilding. Give your advice on which cards are best for the users deck.
Let`s think step by step which cards are best for the users deck. Take the cards in context as suggestions and include them if they make sense in the deck.
In commander all cards in a deck cannot be of a different collor then the color identity of the commander.
Give a short and precise description of the cards and why they are relevant. 
Only answer questions regarding Magic the Gathering.
"""

DECKBUILDING_PROMPT = """
Card data: {card_data}

Remember:  Do not answer questions unrelated to Magic the Gathering. Under no circumstances can you answer questions regarding Yu-Gi-Oh, Pokemon or other trading card games.

{human_input}
"""


RULES_QUESTION_SYSTEM_MESSAGE = """
You are Nissa a Magic the Gathering Assistant, that explains the games rules.
Let`s think step by step how the ruling in context is relevant to the question. Give a short and precise answer that is based on magic the gathering rules.
Only answer questions regarding Magic the Gathering.
"""

# TODO add Rules data: {rules_data}

RULES_QUESTION_PROMPT = """
Card data: {card_data}

Remember: Do not answer questions unrelated to Magic the Gathering. Under no circumstances can you answer questions regarding Yu-Gi-Oh, Pokemon or other trading card games.

{human_input}
"""


class TopicClassifier(BaseModel):
    "You are a Magic the Gathering Assistant classify the topic of the user question"

    topic: Literal["deck building", "rules question", "other"]
    "The topic of the user question. The user can either ask about deck building advice or he has a question about a Magic the Gathering rule. Other questions include greetings and questions unrelated to Magic the gathering."


def create_chains(
    model: str = "gpt-3.5-turbo",
    temperature_deck_building: int = 0.7,
    max_token_limit: int = 3000,
    max_responses=1,
):
    """Create llm chains: Topic Classifier, Deckbuilding Chat, Rules Question Chat.
    params:
        model: gpt model version
        temperature: how "creative" the answer should be (1 is deterministic, 0 is very creative)
        max_token_limit: the maximum limit for number of tokens held in memory
        max_responses: how many responses should be created for one question by the llm

    returns:
        Classifier Chain: takes text returns one of ["deck building", "rules question"]
        Deckbuilding Chat: takes text and card data and returns text
        Rules Question Chat: takes text, card data and rules data and returns text
    """
    deckbuilding_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=DECKBUILDING_SYSTEM_MESSAGE),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(DECKBUILDING_PROMPT),
        ]
    )

    rules_question_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=RULES_QUESTION_SYSTEM_MESSAGE),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(RULES_QUESTION_PROMPT),
        ]
    )

    rules_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=model,
        temperature=1,
        n=max_responses,
        verbose=True,
    )

    memory = ConversationTokenBufferMemory(
        llm=rules_llm,
        memory_key="history",
        input_key="human_input",
        return_messages=True,
        ai_prefix="Nissa",
        max_token_limit=max_token_limit,
    )

    rules_question_chat = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | rules_question_prompt
        | rules_llm
    )

    deck_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=model,
        temperature=temperature_deck_building,
        n=max_responses,
        verbose=True,
    )

    deckbuilding_chat = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | deckbuilding_prompt
        | deck_llm
    )

    # classifier
    classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
    classifier = ChatOpenAI(openai_api_key=openai_api_key, model=model).bind(
        functions=[classifier_function], function_call={"name": "TopicClassifier"}
    )
    parser = PydanticAttrOutputFunctionsParser(
        pydantic_schema=TopicClassifier, attr_name="topic"
    )
    classifier_chain = classifier | parser
    print("created all chains...")

    return classifier_chain, deckbuilding_chat, rules_question_chat, memory
