import openai

# Set up your OpenAI API key
openai.api_key = "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"

def get_person_details(name):
    # Define the initial conversation prompt
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that provides details about people."},
        {"role": "user", "content": f"What can you tell me about {name}?"}
    ]

    # Generate the response using the Chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Extract and return the assistant's reply
    reply = response.choices[0].message['content']
    return reply

# Get the name from the user
name = input("Enter the name of the person: ")

# Get the person details using the conversational agent
details = get_person_details(name)

# Print the details
print(details)
