import openai
import creds
openai.api_key = creds.api_key

def summarize_data(article):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "you summarize the whole thing that the user sends in 3 sentences"},
            {"role": "user", "content": article}
        ]
    )
    #completion.choices[0].message
    return completion.choices[0].message["content"]