from consts import openai
import json
from weather_api_call import get_current_weather
from finance_app import stock_monthly_performance


# Step 1, send model the user query and what functions it has access to
def run_conversation(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": str({query})}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "stock_monthly_performance",
                "description": "Display like MSFT, and Get the stock price and analyze it is put or call",
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
            },
        ],
        function_call="auto",
    )

    # print(response)

    message = response["choices"][0]["message"]

    # print("\n\n\n", message, "Here is the message")

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        available_functions = {
            "get_current_weather": get_current_weather,
            "stock_monthly_performance": stock_monthly_performance,
        }
        function_name = message["function_call"]["name"]

        function_to_call = available_functions[function_name]
        function_args = json.loads(message["function_call"]["arguments"])

        # Step 3, call the function
        # Note: the JSON response from the model may not be valid JSON
        function_response = function_to_call(**function_args)

        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": str({query})},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )

        return second_response["choices"][0]["message"]

    return message


def main():
    with open("user_input.log", "w") as file:
        while True:
            query = input("\nEnter a question: ")

            # Write the query to the log file
            file.write(query + "\n")

            if query == "quit":
                break

            answer = run_conversation(query)

            # Print the answer
            print(f"\n\n > Question:")
            print(query)
            print(f"\n\n > Answer:")
            print(answer)


if __name__ == "__main__":
    main()
