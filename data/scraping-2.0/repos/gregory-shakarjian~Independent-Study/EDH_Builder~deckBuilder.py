from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chat_models import ChatOpenAI

import os
os.environ['OPENAI_API_KEY'] = ''
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == "__main__":
    chat = ChatOpenAI(openai_api_key=api_key)
    
    deck_rules = "Players choose a legendary creature as the commander for their deck. A card’s color identity is its color plus the color of any mana symbols in the card’s rules text. A card’s color identity is established before the game begins, and cannot be changed by game effects. The cards in a deck may not have any colors in their color identity which are not in the color identity of the deck’s commander. A Commander deck must contain exactly 100 cards, including the commander. With the exception of basic lands, no two cards in the deck may have the same English name. Some cards (e.g. Relentless Rats) may have rules text that overrides this restriction."
    deck_rules += "Include the following amount of cards: 15 basic lands, 22 other lands, 10 mana ramp, 10 card draw, 6 targeted removal, 4 board wipes, and 32 other cards that synergize with the Commander's abilities."

    format_rules = "Your response should be a list of new line seperated values, eg: 'foo  \nbar  \nbaz"

    commander = input("\nCommander? ")

    budget = input("\nWhat is your budget? ")

    system_template="You are a deck builder for the Magic the Gathering game mode Commander with a budget of {budget_value} dollars. Your job is to create a competitive deck within your budget referencing these rules: {rules}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template="{commander_name}\n{format_instructions}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    request = chat_prompt.format_prompt(budget_value = budget, commander_name = commander, rules = deck_rules, format_instructions = format_rules).to_messages()

    result = chat(request)

    print(result.content)