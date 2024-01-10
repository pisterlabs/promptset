import os

from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from langchain.output_parsers import PydanticOutputParser

load_dotenv(find_dotenv())

api_key = os.getenv("ANTHROPIC_API_KEY")
chat = ChatAnthropic(anthropic_api_key=api_key, model="claude-2")


class AnkiCard(BaseModel):
    front: str
    back: str


class Deck(BaseModel):
    name: str
    cards: List[AnkiCard]


def create_card(subtopic):
    parser = PydanticOutputParser(pydantic_object=AnkiCard)
    prompt = PromptTemplate(
        template="Give me an anki card with a few sentences for the subtopic with the question in the front, the answer in the back \n{format_instructions}\n{subtopic}\n",
        input_variables=["subtopic"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(subtopic=subtopic)

    output = chat.predict(_input.text)
    result = parser.parse(output)
    return result


class Topic(BaseModel):
    topic: str
    subtopics: List[str]

class Answer(BaseModel):
    score: int
    explanation: str
    
class FactCheck(BaseModel):
    score: int
    verdict: str
    explanation: str
    possible_changes: List[str]

def create_subtopics(topic, num):
    parser = PydanticOutputParser(pydantic_object=Topic)
    prompt = PromptTemplate(
        template="Give me a list of {num} topics for the topic\n{format_instructions}\n{topic}\n",
        input_variables=["topic", "num"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(topic=topic, num=num)

    output = chat.predict(_input.text)
    result = parser.parse(output)
    return result


def fact_check(card_front: str, card_back: str, evidence: str):
    parser = PydanticOutputParser(pydantic_object=FactCheck)
    prompt = PromptTemplate(
        template="Here is the front of the current anki card: <FRONT>{card_front}</FRONT>\n \
                and here is the back of the current anki card: <BACK>{card_back}</BACK>.\n \
                The back is supposed to be the answer to the front of the card.\
                Here is some expert evidence that you should use to evaluate if the card is correct: <EVIDENCE>{evidence}</EVIDENCE>\
                Does the evidence show that the card is correct? Explain your reasoning by making references to the evidence.\n \
                If there needs to be any corrections, make a list of possible_changes. Don't add punctuation to strings in the json \
                Give a one sentence verdict on if the card needs to be changed, and give a score out of 100 of how well the card is made. \
                \n{format_instructions}\n",
        input_variables=["card_front", "card_back", "evidence"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    _input = prompt.format_prompt(
        card_front=card_front, card_back=card_back, evidence=evidence
    )

    output = chat.predict(_input.text)
    result = parser.parse(output)
    return result

def answer_eval(card_front: str, card_back: str, answer: str):
    parser = PydanticOutputParser(pydantic_object=Answer)
    prompt = PromptTemplate(
        template="Here is the front of the current anki card: <FRONT>{card_front}</FRONT>\n \
                and here is the back of the current anki card: <BACK>{card_back}</BACK>.\n \
                The back is supposed to be the answer to the front of the card.\
                Here is some answer that a I wrote: <ANSWER>{answer}</ANSWER>\
                How do you think the I did?  Explain your reasoning.\n \
                If you have suggestions for improvement, suggest what they should be.\
                    Rate my answer on a scale of 1 to 10, 1 being the worst.\n{format_instructions}\n",
        input_variables=["card_front", "card_back", "answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    _input = prompt.format_prompt(
        card_front=card_front, card_back=card_back, answer=answer
    )

    output = chat.predict(_input.text)
    result = parser.parse(output)
    result = result.dict()
    return result


def create_deck(topic, num_cards):
    subtopics = create_subtopics(topic, num_cards)
    cards = []
    for subtopic in subtopics.subtopics:
        card = create_card(subtopic)
        cards.append(card)

    deck = Deck(cards=cards, name=subtopics.topic)
    return deck


def test_create_deck():
    deck = create_deck("Linear Algebra", 5)
    print(deck)
    
def test_fact_check():
    evidence = "Brian is a 70 year-old alcoholic and needs a new kidney"
    card_front = "Is Brian OK?"
    card_back = (
        "Yeah Brian is OK I saw him yesterday and all his organs seem to be working"
    )
    print(fact_check(card_front, card_back, evidence))


def test_answer_eval():
    card_front = "What is a partial differential equation?"
    card_back = '''A partial differential equation is a type of mathematical equation that involves the partial derivatives\
        of a function with respect to one or more independent variables. Partial derivatives are the rates of change of a \
            function along a specific direction, such as x or y. Partial differential equations are used to model many natural\
                phenomena, such as heat, sound, waves, fluid flow, and electromagnetism.'''
    answer = "Its some sort of math thing, people use it for physical modelling and statistics. It is a type of inequality"
    print(answer_eval(card_front, card_back, answer))


def main():
    test_fact_check()


if __name__ == "__main__":
    main()
