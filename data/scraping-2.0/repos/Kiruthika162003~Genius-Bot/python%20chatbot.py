import openai

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

def main():
    print("Genius Bot: Hello! I'm Genius Bot, an AI-powered chatbot. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Genius Bot: Goodbye!")
            break

        response = generate_response(user_input)
        print("Genius Bot:", response)

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose an appropriate engine
        prompt=prompt,
        max_tokens=50  # You can adjust this to control response length
    )
    
    return response.choices[0].text.strip()

if __name__ == "__main__":
    main()
