from langchain.agents import load_tools, AgentType, initialize_agent
from langchain import PromptTemplate
from langchain.agents import Tool

from langchain.llms import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from serpapi import GoogleSearch
import os
import json

load_dotenv()


def find_price_w_serpapi(name: str) -> {"price": int, "url": str}:
    params = {
        "q": name,
        "hl": "tr",
        "gl": "tr",
        "google_domain": "google.com.tr",
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic_results = results.get("organic_results")
    for result in organic_results:
        try:
            price = (
                result.get("rich_snippet")
                .get("top")
                .get("detected_extensions")
                .get("price")
            )

            url = result.get("link")
            return {
                "price": int(price),
                "url": url,
            }
        except:
            continue

    return "ERROR. STOP IMMEDIATELY."


def core(user_input: str) -> str:
    """Core function of the application. Takes user input and returns a response."""

    llm = OpenAI(temperature=0.2)

    class Find_Price_And_Url(BaseModel):
        name: str = Field(..., description="Name of the book")

    tools = load_tools(
        ["serpapi"],
        llm=llm,
    )

    tools.append(
        Tool.from_function(
            name="Find_Price_And_Url",
            description="Useful when you need to find the price of the book in a website. Use Turkish name of the book.",
            args_schema=Find_Price_And_Url,
            func=find_price_w_serpapi,
        ),
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    bookFinder = PromptTemplate.from_template(
        template="""
            Book: {book}.
            1 - Find the name of turkish version of the book.
            2 - Search the internet with its Turkish name and find a website that sells it.
            3 - Print the author,name,turkish_name, price and url of the book as Python dictionary like below. Make sure it is valid JSON.:
            {{"author": "Author", "name": "Book", "turkish_name": "Turkish version name", "price": 0,"url" : "First url that you find"}}
        """
    )

    prompt = bookFinder.format(book=user_input)

    response = agent.run(prompt)

    return json.loads(response)


# print(core("Atomic Habits"))
