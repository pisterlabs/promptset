# Import the necessary libraries
import openai
import gradio as gr

# Set your OpenAI API key for authentication
openai.api_key = "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"

# Define an initial message to start the conversation with the AI Chat Assistant
messages = [{"role": "system", "content": "AI Chat Assistant for all your need"}]

# Define a function called AIChatAssistant that takes user input and generates a response
def AIChatAssistant(user_input, messages=messages):
    # Append the user input to the existing messages
    messages.append({"role": "user", "content": user_input})

    # Generate a response using the OpenAI Chat Completion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the assistant's reply from the API response
    ChatAssistant_reply = response["choices"][0]["message"]["content"]

    # Append the assistant's reply to the messages
    messages.append({"role": "assistant", "content": ChatAssistant_reply})

    # Return the assistant's reply and updated messages
    all_responses = "\n".join([message["content"] for message in messages])
    return ChatAssistant_reply, all_responses

# Create a Gradio interface for the AIChatAssistant function
def chat_interface():
    text_input = gr.Textbox()
    output_text = gr.Textbox()
    previous_responses = gr.Textbox()

    def chat_assistant_with_ui(user_input):
        assistant_reply, all_responses = AIChatAssistant(user_input)
        previous_responses_obj = gr.Textbox(all_responses)  # Create a new object with updated text
        return assistant_reply, previous_responses_obj

    gr.Interface(fn=chat_assistant_with_ui, inputs=text_input, outputs=[output_text, previous_responses], title="AI Chat Assistant").launch(share=True)

# Launch the Gradio interface with the chat_interface function
chat_interface()
