import openai
import os, sys

# Setup OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize conversation history
conversation_history = []

while True:
    # User question
    print("\nPlease enter your question (or 'exit' to end):")
    question = input()

    if question.lower() == 'exit':
        break

    # Add user question to conversation history
    conversation_history.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        max_tokens=256,
        n=1,
        stop=None,
        temperature=0.7,
        stream=True
    )

    for chunk in response:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)

    # Add API response to converstion history
    conversation_history.append({"role": "system", "content": content})

    
print("Conversation ended.")