import streamlit as st
import requests
from newspaper import Article
from langchain_core.messages import HumanMessage
from langchain.schema import (HumanMessage)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import validator, BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate

class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @validator('summary', allow_reuse=True)
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines
    


def initializeLLM(api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0)
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0)
    return llm


def summarize(article_title, article_text, api_key):
    # set up output parser
    parser = PydanticOutputParser(pydantic_object=ArticleSummary)

    #preparing template for the prompt
    template = """As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:

    Example 1:
    Original Article: 'The Effects of Climate Change
    Summary:
    - Climate change is causing a rise in global temperatures.
    - This leads to melting ice caps and rising sea levels.
    - Resulting in more frequent and severe weather conditions.

    Example 2:
    Original Article: 'The Evolution of Artificial Intelligence
    Summary:
    - Artificial Intelligence (AI) has developed significantly over the past decade.
    - AI is now used in multiple fields such as healthcare, finance, and transportation.
    - The future of AI is promising but requires careful regulation.

    Now, here's the article you need to summarize:
    =====================
    Title: {article_title}

    Text: {article_text}
    =====================

    {format_instructions}
    """

    # prompt = template.format(article_title=article_title, article_text=article_text)
    prompt = PromptTemplate(
        template=template,
        input_variables=['article_title', 'article_text'],
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    # Format the prompt using the article title and text obtained from scraping
    formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text)

    # messages = HumanMessage(content=[
    #     {
    #         'type': 'text',
    #         'text': prompt,
    #     }
    # ])


    #Setting llm
    llm = initializeLLM(api_key)
    response = llm.invoke(formatted_prompt.to_string())

    # Parse the output into the Pydantic model
    parsed_output = parser.parse(response.content)
    return parsed_output



def getArticle( api_key, article_url = "https://www.nytimes.com/2024/01/04/us/claudine-gay-harvard-president-race.html",):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

    session = requests.Session()

    try:
        response = session.get(headers=headers, url=article_url, timeout=10)

        if response.status_code == 200:
            article = Article(article_url)
            article.download()
            article.parse()
            summary = summarize(article.title, article.text, api_key)
            print(summary)
            return [summary.summary, summary.title]
        
        else:
            st.write(f"Failed to fetch article at {article_url}")
            return None

    except Exception as e:
        st.write(e)
        return None