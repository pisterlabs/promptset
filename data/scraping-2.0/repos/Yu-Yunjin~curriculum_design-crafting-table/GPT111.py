import openai

openai.api_key = ""


def call_chat_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=None
    )
    # print(response)
    return response.choices[0].text.strip()


# print(call_chat_gpt("请问你知道山东师范大学吗？"))
