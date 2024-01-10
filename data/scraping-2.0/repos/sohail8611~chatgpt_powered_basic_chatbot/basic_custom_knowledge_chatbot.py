import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")



# open the file in read mode
with open('custom_knowledge.txt', 'r') as file:
    contents = file.read()


def mychatbot(query):
    res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please answer my queries according to the given context \nContext: {}".format(str(contents))},
            {"role": "assistant", "content": "Okay sure!"},
            {"role": "user", "content": query}
        ]
    )
    return res


ans = mychatbot('how many subscribers d dot py have?')
print(ans['choices'][0]['message']['content'])



