import openai

openai.api_key = "YOUR OPENAI API"
model_engine = "text-davinci-002"

while True:
    user_input = input("You: ")

    prompt = f"You: {user_input}\nFriend: "

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.5,
    )

    friend_response = response.choices[0].text
    print("Friend:", friend_response)
