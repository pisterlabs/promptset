# import googletrans

# translator = googletrans.Translator()
# inputText = input("Enter text to translate: ")
# output = translator.translate(inputText, dest="english")
# print(output.text)


import openai

# Set your OpenAI API key
# Replace with your actual API key
api_key = ""

# Function to generate greetings


def generate_greeting(prompt):
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="text-davinci-002",  # Use GPT-3 or GPT-4, depending on your access
        prompt=prompt,
        max_tokens=50,  # You can adjust the length of the generated text
    )

    return response.choices[0].text

# Main function


prompt = "Generate a friendly greeting:'"
greeting = generate_greeting(prompt)
print("Generated Greeting:", greeting)
