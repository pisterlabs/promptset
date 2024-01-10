import openai

# Read the key from the file and store it in a variable
with open("open_ai_key.txt", "r") as f:
    key = f.read()

openai.api_key = key
# Create a chat session with gpt-3.5-turbo model
prompt = "Q: Will that be a good day?\nA:"

# Send the prompt to the model and get the response
response = openai.Completion.create(model="gpt-3.5-turbo", prompt=prompt)

# Print the response
print(response.choices[0].text)

