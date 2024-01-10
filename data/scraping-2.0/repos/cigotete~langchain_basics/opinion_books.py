import json
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from decouple import config
from typing import List

from third_parts.gutendex import scrape_gutendex

from output_parsers import suitability_parser

def opinion_books(page=1, limit=2, topics:List[str]=[]) -> str :
        # Scrape books from gutendex
    gutendex_data = scrape_gutendex(page, limit)

    summary_template = """
        given the {books_information} about each book, I want to know:
        1. what is the book title?
        2. According {suitable_topics} subjects, the book is suitable for a person who loves {suitable_topics} topics?
        Please organize your answer per book.
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=['books_information', 'suitable_topics'],template=summary_template,
        partial_variables={
            "format_instructions": suitability_parser.get_format_instructions()
        },
    )

    openai_api_key = config("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    result = chain.run(books_information=gutendex_data, suitable_topics=topics)
    json_data = f"[{result}]"
    parsed_data = json.loads(json_data)
    return parsed_data

if __name__ == '__main__':
    result = opinion_books(1, 2, ["psychology", "fiction"])
    print(result)
