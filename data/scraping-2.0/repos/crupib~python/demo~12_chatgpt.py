# Import the OpenAI library
import openai


# Set up the OpenAI API client :
openai.api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Set up the model (more models, visit https://beta.openai.com/playground)
model_engine = "text-davinci-003"


# Define a function that sends a message to ChatGPT
def chat_query(prompt):
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=2048,
        n=1,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message


# Define a function that handles the conversation
def conversation_handler(prompt):
    # Send the prompt to ChatGPT
    response = chat_query(prompt)
    print(f"ChatGPT: {response}")
    
    # End the conversation if ChatGPT says goodbye
    if ("goodbye") in response.lower():
        return
    
    # Otherwise, get user input and continue the conversation
    prompt = input("Ask your question?: ")
    conversation_handler(prompt)


# Main program starts here:
if __name__ == "__main__":
  # Example of prompt to query
  prompt = "GPT-3 vs ChatGPT: What is the difference?"

  # Start the conversation
  conversation_handler(prompt)
