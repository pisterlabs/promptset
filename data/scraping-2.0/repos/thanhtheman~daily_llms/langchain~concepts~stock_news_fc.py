from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import requests, os, json
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

user_request = """please identify the top stock movers today, look for the news that moved these stocks, and summarize the news to explain why each of these stock moved. 
Please follow this format: company name, percentage of price change and explanation"""

functions = [
    {
        "name": "get_stock_movers",
        "description": "Get the stocks that has biggest price/volume moves, e.g. actives, gainers, losers, etc.",
        "parameters": {
            "type": "object",
            "properties": {
            },
        }
    },
    {
        "name": "get_market_news",
        "desscription": "Get the latest news that moves the stocks",
        "parameters": {
            "type": "object",
            "properties": {
                "performanceIDs": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "a performanceID in a specific format such as '0P000003B1' of a stock"
                    },
                    "description": "a list of all performanceIDs"
                }
            },
            "required" : ["performanceIDs"]
        }
    }
]

def get_stock_movers():
    url = "https://morning-star.p.rapidapi.com/market/v2/get-movers"
    headers = {
        "X-RapidAPI-Key": os.getenv("X-RAPIDAPI-KEY"),
        "X-RapidAPI-Host": "morning-star.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    movers = []
    for i in data["gainers"][:3]:
        movers.append({i["ticker"], i["name"], i["performanceID"], i["percentNetChange"]})
    return movers

def get_market_news(performanceID):
    url = "https://morning-star.p.rapidapi.com/news/list"

    querystring = {"performanceId":performanceID}

    headers = {
        "X-RapidAPI-Key": os.getenv("X-RAPIDAPI-KEY"),
        "X-RapidAPI-Host": "morning-star.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
  
    return response.json()

def format_news(list_ids):
    list_news = []
    for i in list_ids:
        list_title = []    
        list_news.append({i: list_title})
        for j in get_market_news(i)[:3]:
            list_title.append(j["title"])
    return list_news

first_response = llm.predict_messages([HumanMessage(content=user_request)], functions=functions, function_call="auto")
print(first_response)

#making api call to get data
api_response = first_response.additional_kwargs["function_call"]["arguments"]
api_response = str(get_stock_movers())
#plug in the data to the next response
second_response =llm.predict_messages([HumanMessage(content=user_request),
                                       AIMessage(content=api_response)], functions=functions, function_call="auto")

#turning the list from string to list so that we can iterate through it
list_ids = json.loads(second_response.additional_kwargs["function_call"]["arguments"])
list_ids = list_ids["performanceIDs"]
print(list_ids)

# making api call to get market news
second_api_response = str(format_news(list_ids))

#plug in the data to the finally get the explanation of why the stock move
final_response =llm.predict_messages([HumanMessage(content=user_request),
                                       AIMessage(content=api_response),
                                       AIMessage(content=second_api_response)], functions=functions, function_call="auto")
print(final_response.content)