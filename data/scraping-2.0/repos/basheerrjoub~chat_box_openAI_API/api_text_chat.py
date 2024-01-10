import openai
import tkinter as tk

# Set up the OpenAI API client
openai.api_key = "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"

# Create the main window
root = tk.Tk()
root.title("OpenAI Conversation")

# Create the conversation history text widget
history = tk.Text(root)
history.pack()


# Function to send the user's message to the language model and display the response
def send_message():
    # Get the user's message from the input field
    message = input_field.get()

    # Clear the input field
    input_field.delete(0, tk.END)

    # Send the message to the language model and get the response
    response = (
        openai.Completion.create(
            engine="text-ada-001",
            prompt=message,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
        )
        .get("choices")[0]
        .get("text")
    )

    # Display the user's message and the language model's response in the conversation history
    history.insert(tk.END, "You: " + message + "\n")
    history.insert(tk.END, "Assistant: " + response + "\n")


# Create the input field and submit button
input_field = tk.Entry(root)
input_field.pack()
submit_button = tk.Button(root, text="Submit", command=send_message)
submit_button.pack()


# Run the main loop
root.mainloop()
