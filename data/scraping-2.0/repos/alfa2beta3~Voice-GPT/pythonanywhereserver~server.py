import openai
openai.api_key = "I dont tell"

def process(questions):
    # list models
    models = openai.Model.list()

    # print the first model's id
    print(models.data[0].id)

    # create a chat completion
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": questions}])

    # print the chat completion
    #print(chat_completion.choices[0].message.content)

    answer = chat_completion.choices[0].message.content

    return answer
