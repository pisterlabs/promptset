import uvicorn
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentFinish

from common.utils import Utils
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.agents import tool, initialize_agent, AgentType
import requests

app = FastAPI()
@tool
def search_ticker_by_stockname(stock_name: str) -> int:
    """
    find stock ticker using stock name or company name

    below is the response attributes:
        count integer The total number of results for this request.

        next_url string
        If present, this value can be used to fetch the next page of data.

        request_id string
        A request id assigned by the server.

        results array
        An array of tickers that match your query.

        active boolean
        Whether or not the asset is actively traded. False means the asset has been delisted.

        cik string
        The CIK number for this ticker. Find more information here.

        composite_figi string
        The composite OpenFIGI number for this ticker. Find more information here

        currency_name string
        The name of the currency that this asset is traded with.

        delisted_utc string
        The last date that the asset was traded.

        last_updated_utc string
        The information is accurate up to this time.

        locale enum [us, global]
        The locale of the asset.

        market enum [stocks, crypto, fx, otc, indices]
        The market type of the asset.

        name string
        The name of the asset. For stocks/equities this will be the companies registered name. For crypto/fx this will be the name of the currency or coin pair.

        primary_exchange string
        The ISO code of the primary listing exchange for this asset.

        share_class_figi string
        The share Class OpenFIGI number for this ticker. Find more information here

        ticker string
        The exchange symbol that this item is traded under.

        type string
        The type of the asset. Find the types that we support via our Ticker Types API.

        status string
        The status of this request's response.
    """

    print("executing search_tiker_by_stockname ...")
    url = "https://api.polygon.io/v3/reference/tickers?market=stocks&search={stock_name}&active=true&sort=name&apiKey=_O899h4QYZQiv8p_nB1bzp4xEs7sUGAV"\
        .format(stock_name=stock_name)
    response = requests.get(url)
    return response.json()

@tool
def get_stock_price_by_ticker(tiker_name: str, timespan: str, from_date:str, to_date:str) -> int:
    """
    finding open price, close price, mid prices for a stock.
    input requirements:
        * 2 input dates, which is a time range from yyyy-MM-dd to yyyy-MM-dd)
        * timespan(options are day, hour, week, month, year)
        * tiker name, need to call search_tiker_by_stockname to get tiker name for a stock

    below is the response attributes:
    ticker string
    The exchange symbol that this item is traded under.

    adjusted boolean
    Whether or not this response was adjusted for splits.

    queryCount integer
    The number of aggregates (minute or day) used to generate the response.

    request_id string
    A request id assigned by the server.

    resultsCount integer
    The total number of results for this request.

    status string
    The status of this request's response.

    results array, which contains following:

        c number
        The close price for the symbol in the given time period.

        h number
        The highest price for the symbol in the given time period.

        l number
        The lowest price for the symbol in the given time period.

        n integer
        The number of transactions in the aggregate window.

        o number
        The open price for the symbol in the given time period.

        otc boolean
        Whether or not this aggregate is for an OTC ticker. This field will be left off if false.

        t integer
        The Unix Msec timestamp for the start of the aggregate window.

        v number
        The trading volume of the symbol in the given time period.

        vw number
        The volume weighted average price.

    next_url string
    If present, this value can be used to fetch the next page of data.
    """

    print("executing get_stick_price_by_tiker ...")
    url = "https://api.polygon.io/v2/aggs/ticker/{tiker_name}/range/1/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=120&apiKey=_O899h4QYZQiv8p_nB1bzp4xEs7sUGAV"\
        .format(tiker_name=tiker_name, timespan=timespan, from_date=from_date, to_date=to_date)
    response = requests.get(url)
    return response.json()

tools = [search_ticker_by_stockname, get_stock_price_by_ticker]

chatOpenAI = ChatOpenAI(openai_api_key = Utils.get_openai_key())
set_llm_cache(InMemoryCache())

from langchain.tools.render import format_tool_to_openai_function
llm_with_tools = chatOpenAI.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)


agent_executor = initialize_agent(
    tools,
    chatOpenAI,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/ask/{question}")
async def ask_quest(question: str):
    print("incoming qestion: " + question)
    return agent_executor.run(question)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
