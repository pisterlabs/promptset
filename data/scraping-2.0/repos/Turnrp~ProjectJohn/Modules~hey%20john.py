import openai

openai.api_key = "YOUR_API_KEY"


def run(statement):
    chat = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": statement and statement or "Hey john!"}],
    )

    reply = chat[0].message.content
    print(f"John: {reply}")


# Setup
from sys import argv


def call_function(function_name, variable_received):
    try:
        function_to_call = globals()[function_name]
        function_to_call(variable_received)
    except KeyError:
        print("Unknown function:", function_name)


if __name__ == "__main__":
    if len(argv) < 3:
        print("Usage: script2.py <function> <variable>")
    else:
        function_name = argv[1]
        variable_received = argv[2]
        call_function(function_name, variable_received)
