import tkinter as tk
import openai
import my_secrets
import textwrap

# Set the OpenAI API key
openai.api_key = my_secrets.OPENAI_API_KEY

# Define a function to generate a response from OpenAI
def generate_response(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
        timeout=15,
    )
    reply = response.choices[0].text.strip()
    return reply

# Create an instance of Tkinter window
window = tk.Tk()

# Set window title
window.title("My First Tkinter Window")

# Set window dimensions
window.geometry("400x300")

# Create a label for the input box
input_label = tk.Label(window, text="Enter your message:")
input_label.pack()

# Create the input box
input_box = tk.Entry(window)
input_box.pack()

# Create a button to submit the input
submit_button = tk.Button(window, text="Submit", command=lambda: display_results(generate_response(input_box.get())))
submit_button.pack()

# Bind the <Return> event to the submit button
window.bind('<Return>', lambda event: submit_button.invoke())

# Create a label to display the results
results_label = tk.Label(window, text="")
results_label.pack()

# Define a function to display the results
def display_results(input_text):
    # Wrap the input text to a maximum width of 50 characters
    wrapped_text = textwrap.fill(input_text, width=50)
    # Set the text of the results label to the wrapped text
    results_label.config(text="ChatGPT\n" + wrapped_text, bd=2)
    # Clear the input box
    input_box.delete(0, tk.END)

# Run the event loop
window.mainloop()