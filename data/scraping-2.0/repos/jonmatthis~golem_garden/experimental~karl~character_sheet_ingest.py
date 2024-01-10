from dotenv import load_dotenv

load_dotenv()

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chat_models import ChatOpenAI
from experimental.karl.character_model import DnDCharacter
from langchain.prompts import PromptTemplate
from langchain import LLMChain

llm = ChatOpenAI(temperature=1, model_name="gpt-4")

parser = PydanticOutputParser(pydantic_object=DnDCharacter)
prompt = PromptTemplate(
    input_variables=["character_info"],
    template="retrieve character information from: {character_info}\nformat like this:{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = LLMChain(llm=llm, prompt=prompt)
fixing_parser = OutputFixingParser.from_llm(llm=llm, parser=parser)


def format_character_sheet(character_sheet: str) -> DnDCharacter:
    result = chain.run(character_info=character_sheet)
    result = fixing_parser.parse(result)

    return result


def main():
    character_sheet = '''
    Name: [Insert name here]
    Class: Drug addict
    Level: 5
    Race: Talking Horse
    Background: Former Hollywood Sitcom Star
    Personality Traits: Sarcastic, dry-witted, uses humor to cover up insecurities
    Flaws: Overthinks things, struggles with decision-making, addicted to a magical substance
    Abilities: Speech, enhanced physical and magical abilities while under the influence of magical substance
    Spell Set: Eldritch Blast, Hex, Misty Step, Hold Monster, Summon Greater Demon
    Equipment: Gun
    Voiced By: Will Arnett
    '''

    result = format_character_sheet(character_sheet)

    print(result)


if __name__ == '__main__':
    main()
