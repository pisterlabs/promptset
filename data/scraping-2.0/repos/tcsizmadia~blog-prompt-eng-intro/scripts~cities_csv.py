from langchain.llms import Ollama
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)

ollama = Ollama(model="llama2:7b")
ollama.temperature = 0


def list_cities(country):
    csv_parser = CommaSeparatedListOutputParser()
    system_message = SystemMessagePromptTemplate.from_template(
        """
        Answer with a list of cities only. Keep anything else out of your answer.
        The order does not matter, but the list must contain at least 5 cities.
        If you don't know the answer, just write "N/A".
        """
    )

    human_message = HumanMessagePromptTemplate.from_template(
        "Top tourist destinations in: {country}.\n{format_instructions}"
    )

    chat = ChatPromptTemplate.from_messages([system_message, human_message])
    prompt = chat.format_prompt(
        country=country,
        format_instructions=csv_parser.get_format_instructions(),
    ).to_messages()

    return csv_parser.parse(ollama.invoke(prompt))


# Llama2 is not very good at this task, so we need to clean any extra text
# from the output.
def clean_cities(cities):
    cleaned_cities = []
    for city in cities:
        city = city.split("\n")[-1]
        cleaned_cities.append(city)
    return cleaned_cities


country = input("Country: ")
cities = list_cities(country)

print(clean_cities(cities))
