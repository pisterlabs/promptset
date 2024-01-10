import openai

# Set your OpenAI API key here
openai.api_key = "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"

def change_ai_tone(user_input, current_tone):
    # Define available tones
    available_tones = ["friendly", "professional", "casual"]

    # Check if the user wants to change the AI's tone
    if "change tone" in user_input.lower():
        # Extract the desired tone from the user's input
        for tone in available_tones:
            if tone in user_input.lower():
                current_tone = tone
                break

    # Generate a response based on the current tone
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Generate a friendly response: '{user_input}'",
        max_tokens=100,
    )

    return response.choices[0].text.strip(), current_tone

# Initialize the current tone as "friendly"
current_tone = "friendly"

while True:
    user_input = input("You: ")
    response, current_tone = change_ai_tone(user_input, current_tone)
    print(f"AI ({current_tone} tone): {response}")