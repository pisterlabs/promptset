import openai

# openai.api_key = "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"
#
# #model_id = 'ft:gpt-3.5-turbo-0613:personal::81Tsc1dF'
# model_id = 'gpt-3.5-turbo'

openai.api_key = "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"

model_id = 'ft:gpt-3.5-turbo-0613:personal::82L6L6nv'

# message_content = 'Interpret the following user input and convert it into JSON of the form { "intent": ["string"], ' \
#                   '"device":["string"], "location":[ "string"]} ' \
#                   '.Only return JSON with 2 lines. User input:'


def generate_response(user_input, role="user"):
    array_exit = ["", "Bye ChatGPT", " Bye ChatGPT", "bye", "bye chat", " bye", " see you"]
    if user_input in array_exit:
        return None

    message_history.append({'role': 'system', 'content': 'You are a helpful assistant.'})
    message_history.append({"role": role, "content": f"{user_input}"})
    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=message_history
    )
    response = completion.choices[0].message.content
    print(completion.choices[0].message.content.strip())
    message_history.append({"role": "assistant", "content": f"{response}"})
    return response

message_history = []

while True:
    prompt = input('User:')
    conversation = generate_response(prompt, role="user")
    if conversation is None:
        break
