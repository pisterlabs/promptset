import os
import openai


def gpt3(query):
    openai.api_key = '**Your own OpenAI Key here**'
    response = openai.Completion.create(

        prompt="""
        Alter this promt as you wish
        """,

        engine="davinci",
        temperature=1,
        max_tokens=170,
        top_p=0.6,
        frequency_penalty=0.8,
        presence_penalty=0.3,
        stop=["\n"],
    )
    s_story = response.choices[0].text
    return str(s_story)


query = ''''''
print(".\n.\nGenerated permutation: \n.\n.\n.")


response = gpt3(query)
print(response)
print("\n.\n.\n.\n.\n.")
