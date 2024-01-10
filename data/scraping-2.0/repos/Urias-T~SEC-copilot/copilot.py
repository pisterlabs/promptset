import os
import requests

# For debugging and local experimentation

# import langchain
# langchain.debug=True

from prompts import prompt

import yfinance as yf

from pydantic.v1 import BaseModel, Field

# For debugging and local experimentation

# import langchain
# langchain.debug=True

import yfinance as yf

from pydantic.v1 import BaseModel, Field

from langchain.chat_models import ChatOpenAI

from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser


from langchain.tools import tool
from langchain.agents import Tool, initialize_agent

from memory import create_memory


class CurrentStockPriceInput(BaseModel):
    symbol: str = Field(..., description="The ticker symbol for the company whose stock price is to be checked.")


@tool(args_schema=CurrentStockPriceInput)
def get_current_stock_price(symbol: str) -> str:
    """Call this function to get the current stock price of a company."""
    stock_info = yf.Ticker(symbol)
    current_price = stock_info.info["currentPrice"]

    return f"The current price is USD {current_price}"


def get_response(query, configurations, chat_history):

    model = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key=configurations["openai_api_key"])
    
    def retriever(query):
        dataset_config = {
            "dataset_id": "company",
            "data_types": ["10-K", "10-Q"]
        }

        retrieval_config = {
            "num_context": 6
        }

        url = "https://api.kay.ai/retrieve"

        headers = {"API-KEY": configurations["kay_api_key"]}

        payload = {
            "query": query,
            "dataset_config": dataset_config,
            "retrieval_config": retrieval_config
        }

        response = requests.post(url, headers=headers, json=payload)

        context_list = response.json()["contexts"]

        texts = []

        for i in range(0, len(context_list)):
            text = context_list[i]["chunk_embed_text"]
            texts.append(text)

        chain = RunnableMap({
            "question": lambda x: x["question"],
            "context": lambda x: x["context"]
        }) | prompt | model | StrOutputParser()

        answer = chain.invoke({"question": query,
                                "context": texts})
        
        return answer


    memory = create_memory(chat_history=chat_history)

    retrieval_tool = Tool(
        name="Kay AI Vector Store",

        func=retriever,
        description=("Use this tool when answering questions that relate to a company's SEC filings and financials."),
        return_direct=True
    )

    tools = [get_current_stock_price, retrieval_tool]

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=model,
        max_iterations=3,
        early_stopping_method="generate",
        memory=memory,
        handle_parsing_errors=True
    )

    result = agent(query)

    chat_history.append((query, result["output"]))

    return result["output"], chat_history
