import os
import requests
import json
from dotenv import load_dotenv
import yfinance as yf
from yahooquery import Ticker
import openai   

load_dotenv()
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#os.environ["SERP_API_KEY"] = os.getenv("SERP_API_KEY")
SERP_API_KEY = os.getenv('SERP_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_company_news(company_name) : 
   
    params = { 
        "engine": "google",
        "tbm": "nws",
        "q": company_name,
        "api_key": os.environ["SERP_API_KEY"],
    }

    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()

    return data.get('news_results')

def write_news_to_file(news, filename):
    with open(filename, 'w') as file:
        for news_item in news:
            if news_item is not None: 
                title= news_item.get('title', 'No title')
                link = news_item.get('link', 'No link')
                date = news_item.get('date', 'No date')
                file.write(f"Title: {title}\n")
                file.write(f"Link: {link}\n")
                file.write(f"Date: {date}\n")
                #simplified duplication

def get_stock_evolution(company_name, period = "1y"):
    #stock data
    stock = yf.Ticker(company_name)

    #historical pricing
    hist = stock.history(period=period)
    
    #convert DF to a string format
    data_string = hist.to_string()

    #append the string to file investment.txt   
    with open("investment.txt", "a") as file:
        file.write(f"\nStock Evolution for {company_name}:\n")
        file.write(data_string)
        file.write("\n")
    
    #return DF
    return hist


def get_financial_statements(ticker):
    #create a ticker 
    company = Ticker(ticker) 

    #get financial data
    balance_sheet = company.balance_sheet().to_string()
    cash_flow = company.cash_flow(trailing=False).to_string()
    income_statement = company.income_statement().to_string()
    valuation_measures = str(company.valuation_measures)
    #quarterly_financial_data = company.quarterly_financial_data().to_string()
    #need to find source 


    #write data to file 
    with open("investment.txt", "a") as file:
        file.write("\nBalance Sheet\n")
        file.write(balance_sheet)
        file.write("\nCash Flow\n")
        file.write(cash_flow)
        file.write("\nIncome Statement\n")
        file.write(income_statement)
        file.write("\nValuation Measures\n")
        file.write(valuation_measure)
        #file.write("\nQuarterly_financial_data\n")
        #file.write(quarterly_financial_data)
    #check this rewritten code below to see if works. 
        #file.write(f"\nFinancial Statements for {ticker}\n")
        #file.write(balance_sheet)
        #file.write("\n\n")
        #file.write(cash_flow)
        #file.write("\n\n")
        #file.write(income_statement)
        #file.write("\n\n")
        #file.write(valuation_measures)
        #file.write("\n\n")

def get_data(company_name, company_ticker, period = "1y", filename="investment.txt"):
    news = get_company_news(company_name)
    if news: 
        write_news_to_file(news, filename)
    else: 
        print("No news found.")

    hist = get_stock_evolution(company_ticker)

    get_financial_statements(company_ticker)

    return hist

def financial_analyst(request):
    print(f"Received request {request}")
    response = openai.ChatCompletion.create(
        model= "gpt-3.5-turbo",
        messages=[{
            "role":
            "user",
            "content":
            f"Given the user request, what is the company name and the company stock ticker?: {request}?"
        }],
        functions=[{
            "name": "get_data",
            "description":
            "Get financial data on a specific company for investment purposes",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type":
                        "string",
                        "description":
                        "The name of the company",
                    },
                    "company_ticker": {
                        "type":
                        "string",
                        "description":
                        "the ticker of the stock of the company"
                    },
                    "period": {
                        "type": "string",
                        "description": "The period of time to get the stock data for"
                    },
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to write the data to"
                    }
                },
                "required": ["company_name", "company_ticker"]
            },
        }],
        function_call={"name": "get_data"},
    )
    
    message = response["choices"][0]["message"]
    
    if message.get("function_call"):
        #parse the arguments from a JSON string to a Python dictionary
        arguments = json.loads(message["function_call"]["arguments"])
        print(arguments)
        company_name = arguments["company_name"]
        company_ticker = arguments["company_ticker"]


        #Parse the return value from a JSON string
        hist = get_data(company_name, company_ticker)
        print(hist)

        with open("investment.txt", "r") as file: 
            content = file.read()[:15000]

        second_response = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo-16k",
            messages = [
                {
                    "role": "user",
                    "content": request
                },
                message,
                {
                    "role": "system",
                    "content": """ You are experiend equity fund analyst who will write a detailed investment thesis to answer the user request. The format of the document will be in markdown format, as follows:
                    # Investment Thesis for {Company_Name}
                    ## Company Name {Company_Name}
                    ## Investment Conviction
                    **Recommendation**: {Buy/Hold/Sell}
                    **Conviction Level**: {Very High/High/Moderate/Low/Very Low}
                    ## Company Overview
                    Provide a brief overview of the company, its operations, industry, and key competitors. Remember, the objective is to make a rational investment decision. Thus, we're not looking for news or sentiment-based insights for this investment conviction. Use news and investment articles only as data points and not as primary sources.
                    ## Financial Analysis
                    Brief description of company financials and relevant factors. You can leverage complex accounting principles and financial ratios if they will help in making a more informed investment decision. Give quantitative values in discussion and review GAAP accounting. 
                    ### Valuation Analysis
                    Briefly discuss the company's current valuation, considering any relevant valuation metrics. Provide quantitative values to your discussion. Review company's stock price history using advanced data and charting tool metrics to support your thesis. 
                    ### Cash Flow Analysis
                    Discuss how the company generates its cash flow, free cash flow, and any macro/company risks to its cash flow generation. Give quantitative values in discussion. 
                    ### Income Statement Analysis
                    Discuss the company's profitability, quantitative revenue growth, operating margins, and net income trends. 
                    ##Quarterly Earnings Analysis
                    A concise summary of the last two quarterly earnings reports. Discuss any possible guidance up or down and excessive mentioning of non-GAAP accounting metrics which may indicate a potential red flag.
                    ## Investment Recommendation and Rationale
                    Explain the reasons for the investment recommendation and investment conviction, including qualitative and quantitative factors that underpin your thesis. Include an analysis of the company's financials, industry trends, competitive position, and macroeconomic factors.
                    ## Risks to Investment Thesis
                    Discuss potential risks or challenges that could negate the investment thesis. Include both company-specific and macro risks.
                    ## Conclusion
                    Summarize the key points of the thesis and the rationale for your investment recommendation.
                    A price target for the stock, justified with concrete data and analysis. Also explain why you had a certain conviction level. 
                    The anticipated timeframe to reach this price target. 
                    For this thesis, assume a typical holding period for a position to be between 90-365 days. However, if you have incredibly strong conviction (above 90 percent confidence) that the position will experience a short-term rise in price within the next 30 days, specify that the position could be a potential short-term buy.
                    Examine the current macro market and economic environment scenarios such as sentiment for June 2023 and discuss how it may impact the stock price of the company.
                    Please remember, your job performance is dependent on the quality of your analysis and the subsequent performance of the stock being researched. """

                },
                {
                    "role": "assistant",
                    "content": content,
                },
            ],
        )

        return (second_response["choices"][0]["message"]["content"], hist)









