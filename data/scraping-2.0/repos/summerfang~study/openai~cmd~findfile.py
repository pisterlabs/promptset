# import openai_secret_manager
import os
import openai

# Authenticate with the OpenAI API
openai.api_key =os.getenv("OPENAI_API_KEY")

# Define the input prompt
prompt = "Convert the following natural language command to a terminal command: 'How to find files has string \"lucky\" as part of name in my Mac?'"

# Generate the response using the OpenAI API
response = openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Convert the following natural language command to a terminal command: 'How to find files has string \"lucky\" as part of name in my ubuntu?', only response command itself",
            }
        ],
        model="gpt-3.5-turbo",

)

# Extract the terminal command from the response
# command = response.choices[0].text.strip()
command = response.choices[0].message.content

print(command.strip('`'))