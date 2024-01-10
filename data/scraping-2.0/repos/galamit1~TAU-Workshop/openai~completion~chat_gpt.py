import openai

# TODO add api keys:
# openai.organization = ""
# openai.api_key = ""


# Define a function to send a message to ChatGPT and get a response
def chat_with_gpt(initial_instructions, prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": initial_instructions},
            {"role": "user", "content": prompt}
        ],
        n=1,
        max_tokens=150,
    )

    return completion.choices[0].message["content"]



