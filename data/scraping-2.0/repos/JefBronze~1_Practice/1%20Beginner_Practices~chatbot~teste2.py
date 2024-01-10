import openai

# Ensure you have set the OPENAI_API_KEY in your environment variables
openai.api_key = 'OPENAI_API_KEY'

def ask_gpt4(question):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that knows a lot about oil prices."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    # Extract the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']
    
    return assistant_message

# Test the function
response = ask_gpt4("What was the average oil price in the 80's?")
print(response)
