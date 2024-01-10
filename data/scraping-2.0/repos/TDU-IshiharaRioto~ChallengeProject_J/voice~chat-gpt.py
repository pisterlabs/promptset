import openai
openai.api_key_path = "C:\\Users\\Makoto\\Desktop\\chat-gpt-key"

question = "何か遅れている路線はある？。"

functions = [
    {
        "name": "train_info",
        "description": "Display train operation information when asked for train operation information.If no route name is specified, 'all'.",
        "parameters": {
            "type": "object",
            "properties": {
                "railway line": {"type": "string"},
            },
            "required": ["railway line"],
        },
    },
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    functions=functions,
    messages=[
        {"role": "user", "content": question},
    ],
)

print(response)
try:
    print(response.choices[0]["message"]["function_call"]["arguments"])
except KeyError:
    print(response.choices[0]["message"]["content"])