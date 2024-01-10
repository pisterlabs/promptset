import openai

# Go to https://platform.openai.com/account/api-keys 
# and generate your api key and paste it inside a 
# file in this directory named "openai_api_key"
key = open("openai_api_key")
openai.api_key = key.read()
q = input("Enter question: ")

response = openai.ChatCompletion.create(model='gpt-3.5-turbo', 
    messages=[{"role":"user", "content": q}])

# Extract and print the generated response
print(response.choices[0].message.content)
