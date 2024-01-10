import os

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_TEMERATURE = 1.1


def generate_movie_title(name: str = "Rob Zombie"):
    llm = OpenAI(temperature=MODEL_TEMERATURE)
    prompt_template_director = PromptTemplate(
        input_variables=["director_name"],
        template="Generate 3 movie titles that would fit director's {director_name}'s next movie",
    )

    title_chain = LLMChain(llm=llm, prompt=prompt_template_director)

    response = title_chain({"director_name": name})

    return response


if __name__ == "__main__":
    print(generate_movie_title(name="Mark Stewartny"))
