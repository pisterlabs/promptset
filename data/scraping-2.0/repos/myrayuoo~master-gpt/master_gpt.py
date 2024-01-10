import csv
import random
import openai

# Set up your OpenAI API credentials
openai.api_key = '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'

# Load the CSV file
def load_options(): 
    options = []
    with open('options.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            options.append(row)
    return options

# Generate a response using ChatGPT
def generate_response(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()

# Main menu loop
def menu_loop(options):
    while True:
        print("\n--- Main Menu ---")
        for i, option in enumerate(options):
            print(f"{i+1}. {option['act']}")
        print("0. Exit")

        choice = input("Enter your choice: ")
        if choice == "!menu":
            continue

        if choice == "0":
            break

        choice = int(choice) - 1
        if choice < 0 or choice >= len(options):
            print("Invalid choice. Please try again.")
            continue

        selected_option = options[choice]
        prompt = selected_option['prompt']
        print(f"\nPrompt: {prompt}")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "!menu":
                break

            prompt += f"\nUser: {user_input}"
            response = generate_response(prompt)
            prompt += f"\nChatGPT: {response}"
            print(f"ChatGPT: {response}")

# Load the options from the CSV file
options = load_options()

# Start the menu loop
menu_loop(options)
