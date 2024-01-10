import openai
import gradio

# Replace "API-key" with your actual OpenAI API key
openai.api_key = "API-key"

# Initial system message
messages = [{"role": "system", "content": "You are an Enthusiastic and Talented Computer Science Student"}]

# Function to interact with the GPT-3.5 Turbo model
def CustomChatGPT(user_input):
    # Append user input to the messages
    messages.append({"role": "user", "content": user_input})
    
    # Request a response from the GPT-3.5 Turbo model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    # Extract the assistant's reply from the GPT-3.5 Turbo response
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    
    # Append the assistant's reply to the messages
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    
    # Return the assistant's reply
    return ChatGPT_reply

# Additional comments to make the code longer
# This is a simple chat interface using Gradio and OpenAI GPT-3.5 Turbo
# The following lines create a Gradio interface for the chat
demo = gradio.Interface(fn=CustomChatGPT, inputs="text", outputs="text", title="ResumeWhisper AI")

# Launch the Gradio interface
# The following line launches the Gradio interface and allows sharing
demo.launch(share=True)

# Additional unnecessary variable to make the code longer
extra_variable = "This variable serves no actual purpose in this script."

# Print the unnecessary variable
print(extra_variable)
