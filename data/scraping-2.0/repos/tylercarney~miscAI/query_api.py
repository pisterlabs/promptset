import openai
import keyring

# Retrieve API key from the system keyring
openai.api_key = keyring.get_password("system", "openai_api_key")

content="""
Whatever you want
can go here and will be submitted as your
prompt.
"""


response = openai.ChatCompletion.create(
    model="gpt-4-0613", 
    messages=[{"role": "user", "content": content}]
    )    
generated_code = response.choices[0].message.content
print(generated_code)
