def ask_assistant(transcription: str) -> str:
    import openai, os, json
    from assistant.openai_functions import (
        function_descriptions,
        available_functions,
    )

    final_answer = ""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You only understand English and Spanish. If you don't understand something, you can ask the user to repeat the sentence.",
        },
        {"role": "user", "content": "Hello, who are you?"},
        {
            "role": "assistant",
            "content": "I am an AI created by OpenAI. How can I help you today?",
        },
        {"role": "user", "content": transcription},
    ]

    chat = openai.ChatCompletion.create(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=function_descriptions,
        function_call="auto",
    )

    response_message = chat["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = available_functions

        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(function_args)

        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )

        final_answer = second_response["choices"][0]["message"]

    else:
        final_answer = response_message

    return final_answer["content"]
