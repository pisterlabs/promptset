import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = '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'

# Initialize the OpenAI API client
openai.api_key = api_key

def chat_with_gpt3(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Specify the GPT-3.5 Turbo model
            messages=[
                {"role": "system", "content": "You are a online safety expert, helping users stay safe online."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message["content"].strip()
    except Exception as e:
        return str(e)

print("Online Safety Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Online Safety Chatbot: Goodbye! Stay safe online!")
        break

    # Get the AI's response
    ai_response = chat_with_gpt3(user_input)

    print("Online Safety Chatbot:", ai_response)
