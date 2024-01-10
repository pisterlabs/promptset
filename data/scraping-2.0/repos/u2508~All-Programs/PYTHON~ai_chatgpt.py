import openai as ai
ai.api_key = "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"

def generate_response(prompt):
    response = ai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1440,
        n=1,
        stop=None,
        temperature=0.6,
    )
    message = response.choices[0].text.strip()
    return message

print("Chatbot: Hi, how can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "goodbye"]:
        print("Chatbot: Goodbye!")
        break
    prompt = f"Conversation\nUser: {user_input}\nChatbot:"
    response = generate_response(prompt)
    print(f"Chatbot: {response}")
