import openai
import json

Model = "gpt-3.5-turbo-0613"
openai.api_key = "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"
MAX_TOKEN = 50

def get_nutrition(calories, fats, protein):
    sample = {
        "calories":calories,
        "fats":fats,
        "protein":protein,
    }
    return json.dumps(sample)

available_functions = {
    "get_nutrition":get_nutrition,
}

def chat_ai(model, messages, max_token):
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        #max_tokens = max_token,
    )
    #return response["choices"][0]["message"]
    return response


def call_function_ai(model, messages, functions, function_call, max_token):
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        functions = functions, 
        function_call = "auto",
        max_tokens = max_token,
    )
    print(response)
    return response["choices"][0]["message"]

def update_messages_response(messages, response_message):
    messages.append(response_message)
    return messages

def update_messages_function_response(messages, name, function_response):
    messages.append(
        {
            "role":"function",
            "name": name,
            "content":function_response,
        }
    )
    return messages

def call_function(response_message, available_functions):
    if response_message.get("function_call"):
        function_call = response_message["function_call"]
        function_name = function_call["name"]
        function_args = json.loads(function_call["arguments"])
        function_to_call = available_functions[function_name]

        function_response = function_to_call(
            calories = function_args.get("calories"),
            fats = function_args.get("fats"),
            protein = function_args.get("protein")
        )

        return function_response, function_name
    print("This is empty")

def generate_function():
    functions = [
        {
            "name":"get_nutrition",
            "description":"get food nutrition",
            "parameters": {
                "type":"object",
                "properties": {
                    "calories": {
                        "type":"string",
                        "description":"amount of calories",
                    },
                    "fats": {
                        "type":"string",
                        "description": "amount of fats in grams",
                    },
                     "protein": {
                        "type":"string",
                        "description": "amount of protein in grams",
                    },
                },
            "required":["calories", "fats", "protein"]
            },
        }
    ]
    return functions

def generate_initial_message():
    return [{"role":"user", "content":"Get the estimate nutrition value of sate ayam in sate khas senayan"}]

def run_conversation():
    messages = generate_initial_message()
    functions = generate_function()
    first_response = call_function_ai(Model, messages, functions, "auto", MAX_TOKEN)
    print(first_response)
    function_response, function_name = call_function(first_response, available_functions)
    messages = update_messages_response(messages, first_response)
    messages = update_messages_function_response(messages, function_name, function_response)
    second_response = chat_ai(Model, messages, MAX_TOKEN)
    return second_response


run_conversation()

# messages = [
#     {
#         "role":"user",
#        "content":"Get the estimate nutrition value of sate ayam in sate khas senayan",
#    },
#   {
#       "role":"function",
#      "name":"get_nutrition",
#   "content":"{\"calories\": \"100\", \"fats\": \"10\", \"protein\": \"20\"}",
# },    