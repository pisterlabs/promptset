import asyncio
import os
import chainlit as cl
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate
)
from langchain.llms import OpenAI
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

examples = [
    {"word": "happy",     "antonym": "sad"},
    {"word": "content",   "antonym": "dissatisfied"},
    {"word": "peaceful",  "antonym": "belligerent"},
    {"word": "tall",      "antonym": "short"},
    {"word": "high",      "antonym": "low"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "fast",      "antonym": "slow"},
    {"word": "sunny",     "antonym": "gloomy"},
    {"word": "clear",     "antonym": "cloudy"},
    {"word": "windy",     "antonym": "calm"},

]

example_formatter_template = \
"""
Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

@cl.on_message # for every user message
def main(input_word: str):

    example_selector = SemanticSimilarityExampleSelector.from_examples (
        examples,
        # Class to create embeddings 
        OpenAIEmbeddings(),
        # VectorStore class to store embeddings and do similarity search
        Chroma,
        # Number of examples to produce
        k=2
    )
    
    fewshot_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input word",
        suffix="Word: {word}\nAntonym:", 
        input_variables=["word"],
    )

    llm = OpenAI(openai_api_key=OPENAI_API_KEY,
                    model=MODEL_NAME)
    response = llm(fewshot_prompt.format(word=input_word))
    
    response += "\n\n=> Enter a word:"

    # final answer
    asyncio.run(
        cl.Message(
            content=response
        ).send()
    )
        
@cl.on_chat_start
def start():
    output = ""
    for e in examples:
        output += f"word: {e['word']} <=> "
        output += f"antonym: {e['antonym']}\n"
        
    output += "\n\n=> Enter a word:"
    asyncio.run(
        cl.Message(
            content=output
        ).send()
    )
