import openai

# You should get your API key from the OpenAI website.
with open('api_key.txt', 'r') as file:
    openai.api_key = file.read().strip()

user_input = input("Hello! I'm your virtual realtor. Tell me more about what you are looking for: ")

    
# Start the conversation
conversation = [
    {"role": "system", "content": "You are a realtor who assists buyers in finding properties that match their criteria and preferences. They provide information on available properties, schedule property viewings, and guide buyers throughout the purchasing process."},
    {"role": "user", "content": user_input}
]

# Let's use the OpenAI GPT-3 model to generate some text.
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=conversation,
  max_tokens=60
)

print(response['choices'][0]['message']['content'])