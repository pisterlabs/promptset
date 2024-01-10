import yfinance as yf
import openai
import json
from consts import openai


def stock_monthly_performance(stock_name, review_period="1mo"):

    stock = yf.Ticker(stock_name)
    data = stock.history(period=review_period)

    data = data[["Open", "High", "Low", "Volume"]]

    return data.to_json()


def stock_trader(query):

    # send model to the user query and what functions it has access to
    model = "gpt-3.5-turbo-0613"
    messages = [{"role": "user", "content": str({query})}]
    functions = [
        {
            "name": "stock_monthly_performance",
            "description": "Get the stock price and analyze it is put or call",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_name": {
                        "type": "string",
                        "description": "the name of the stock to analyze",
                    },
                    "review_period": {"type": "string", "enum": ["1mo", "3mo"]},
                },
                "requred": ["stock_name"],
            },
        }
    ]

    response = openai.ChatCompletion.create(
        model=model, messages=messages, functions=functions, function_call="auto"
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {"stock_monthly_performance": stock_monthly_performance}

        function_name = response_message["function_call"]["name"]

        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])

        # Step3, call the function
        # Note: the JSON response from the model may not be valid JSON
        function_response = function_to_call(**function_args)

        # Step4: send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": str({query})},
                response_message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )

        return second_response["choices"][0]["message"]

    return response_message


def main():
    with open("user_input.log", "w") as file:
        while True:
            query = input("\nEnter a question: ")

            # Write the query to the log file
            file.write(query + "\n")

            if query == "quit":
                break

            answer = stock_trader(query)

            # Print the answer
            print(f"\n\n > Question:")
            print(query)
            print(f"\n\n > Answer:")
            print(answer)


if __name__ == "__main__":
    main()
