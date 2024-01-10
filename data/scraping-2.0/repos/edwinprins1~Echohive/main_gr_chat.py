import sys
import openai
import tiktoken
import gradio as gr

# Define the function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# Set the maximum token limit
MAX_MEMORY_TOKENS = 100

# Initialize the conversation_history list
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def chatbot(chat_input):
    global conversation_history

    # Append user input to the conversation history
    conversation_history.append({"role": "user", "content": chat_input})
    
    # Calculate the total tokens in the conversation history
    total_tokens = sum(count_tokens(message["content"]) for message in conversation_history)

    # Remove the oldest message from conversation history if total tokens exceed the maximum limit
    while total_tokens > MAX_MEMORY_TOKENS:
        if len(conversation_history) > 2:
            removed_message = conversation_history.pop(1)
            total_tokens -= count_tokens(removed_message["content"])
        else:
            break

    # Make API calls to OpenAI with the conversation history and use streaming responses
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation_history,
        stream=True,
    )

    # Process the response from the API
    assistant_response = ""
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue
        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            assistant_response += r_text

    # Append the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Format conversation history for display
    formatted_conversation = ""
    for message in conversation_history:
        if message["role"] == "user":
            formatted_conversation += f'<p><b>You:</b> {message["content"]}</p>'
        elif message["role"] == "assistant":
            formatted_conversation += f'<p><b>Assistant:</b> {message["content"]}</p>'

    return formatted_conversation

# Gradio app
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.inputs.Textbox(lines=2, label="You:"),
    outputs=gr.outputs.HTML(label="Conversation:"),
    title="Chatbot",
    description="A chatbot using OpenAI's GPT-4.",
    allow_screenshot=False,
)

iface.launch()
