import openai

def main():
    # Set your OpenAI API key here
    openai.api_key = "YOUR OPEN AI API KEY"

    while True:
        # Get user input.
        user_input = input("You: ")

        # Use the ChatGPT API to generate a response.
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose the engine you want to use
            prompt=user_input,
            max_tokens=50  # You can adjust this value to control response length
        )

        # Print the response from the model.
        print("Chatbot:", response.choices[0].text.strip())

if __name__ == "__main__":
    main()
