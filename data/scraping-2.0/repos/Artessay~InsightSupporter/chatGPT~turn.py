import os
import openai

# Set up OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")
model_engine = "text-davinci-002"

# Define function to retrieve OpenAI response to customer query
def get_openai_response(query, conversation_context=None):
    if conversation_context is None:
        conversation_context = []
    prompt = "Customer: " + query + "\n" + "\n".join(conversation_context) + "\nChatbot:"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = response.choices[0].text.strip()
    return message

# Implement self-service customer service loop
print("Welcome to our self-service customer service.")
conversation_context = []
while True:
    query = input("Customer: ")
    response = get_openai_response(query, conversation_context)
    conversation_context.append("Customer: " + query)
    conversation_context.append("Chatbot: " + response)
    print("Chatbot:", response)
