import openai

# use your OpenAI account
openai.api_key = "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"

# greet the user
print("Bot: Hello!")

while True:
    # capture user input
    user_input = input("You: ")

    # build a prompt for GPT-3
    prompt = "August is a kind bot who wants to be friends."
    prompt += "\n"
    prompt += "Bot: Hello!"
    prompt += "You: " + user_input
    prompt += "Bot:"

    # send the GPT3 request
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        stop="You:"
    )

    # extract the response
    bot_response = completion.choices[0].text.strip()

    # display the response
    print("Bot: " + bot_response)
