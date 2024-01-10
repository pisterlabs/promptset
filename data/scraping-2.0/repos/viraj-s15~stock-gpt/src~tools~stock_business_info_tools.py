import yfinance as yf
from args import BusinessInfoArgs, CompanyNewsArgs, ContactInfoArgs, OfficerInfoArgs
from langchain.tools import tool


def get_ticker(company_name) -> str:
    """
    Provides the ticker for a given company name
    """
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={"User-Agent": user_agent})
    data = res.json()
    company_code = data["quotes"][0]["symbol"]
    return company_code


@tool("company_news", args_schema=CompanyNewsArgs)
def get_company_news(stock_name: str) -> str:
    """
    Provides news about the company
    """
    try:
        ticker = get_ticker(stock_name)
        if isinstance(ticker, str):
            return ticker
        res = ticker.news
        output = ""
        for i in res:
            output += i["title"] + i["publisher"] + i["link"] + "\n"
        return output
    except Exception as e:
        return f"Error getting company news: {str(e)}"


@tool("contact_information", args_schema=ContactInfoArgs)
def get_contact_information(stock_name: str) -> str:
    """
    Provides contact information of the company
    """
    try:
        ticker = get_ticker(stock_name)
        if isinstance(ticker, str):
            return ticker
        info = dict()
        info["Address"] = ticker.info["address1"]
        info["City"] = ticker.info["city"]
        info["State"] = ticker.info["state"]
        info["Zip Code"] = ticker.info["zip"]
        info["Phone Number"] = ticker.info["phone"]
        info["Website"] = ticker.info["website"]
        return str(info)
    except Exception as e:
        return f"Error getting contact information: {str(e)}"


@tool("business_info", args_schema=BusinessInfoArgs)
def get_business_info(stock_name: str) -> str:
    """
    Provides general data of its stock including current price, open, close, etc
    """
    try:
        ticker = get_ticker(stock_name)
        if isinstance(ticker, str):
            return ticker  # Return error message
        info = dict()
        info["Industry"] = ticker.info["industry"]
        info["Sector in Industry"] = ticker.info["sectorDisp"]
        info["About"] = ticker.info["longBusinessSummary"]
        return str(info)
    except Exception as e:
        return f"Error getting business information: {str(e)}"


@tool("officer_info", args_schema=OfficerInfoArgs)
def get_officer_info(stock_name: str) -> str:
    """
    Function which gets info about the officers i.e. CEO, CTO and so on
    """
    try:
        ticker = get_ticker(stock_name)
        if isinstance(ticker, str):
            return ticker  # Return error message
        res = ticker.info["companyOfficers"]
        return str(res)
    except Exception as e:
        return f"Error getting officer information: {str(e)}"


stock_business_tools = [
    get_company_news,
    get_contact_information,
    get_business_info,
    get_officer_info,
]
