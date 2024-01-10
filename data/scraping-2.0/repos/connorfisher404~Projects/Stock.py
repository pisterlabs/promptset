import yfinance as yf
import openai
import json
openai.api_key = 'openai api key'

def get_stock_price(ticker):
    stock = yf.Ticker(ticker)    
    hist = stock.history(period="1d")
    
    return hist['Close'].iloc[0]







def run_conversation():    
    stock = input("Please enter the company you want to know the price of: ")
    messages = [{"role": "user", "content": f"What is the current price of {stock}?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol of a company",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The ticker symbol of the company (e.g. MSFT for Microsoft)",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["ticker"],
                },
            },
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        
        available_functions = {
            "get_stock_price": get_stock_price,
        }  
        messages.append(response_message)  
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                ticker=function_args.get("ticker"),
                
            )
            if not isinstance(function_response, str):
                function_response = str(function_response)

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  
        second_response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  
        return second_response
    
    
response = run_conversation()


message = response.choices[0].message.content


print(message)