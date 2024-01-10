import tkinter as tk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import openai

nltk.download('punkt')
nltk.download('stopwords')

def summarize_paragraph(paragraph):
    # Prompt for ChatGPT to summarize the input paragraph with a positive spin on sustainability
    prompt = "Please summarize this text below with a positive spin on sustainability:\n\"" + paragraph + "\""

    # Call ChatGPT to generate the summary
    summary = chat_gpt(prompt)

    return summary

def chat_gpt(prompt):
    # Set up OpenAI API credentials
    openai.api_key = 'we need an API key lol'

    # Define the parameters for the ChatGPT API call
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Extract the generated summary from the API response
    summary = response.choices[0].text.strip()

    return summary

def run_summarizer():
    paragraph = input_box.get("1.0", tk.END)
    summary = summarize_paragraph(paragraph)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, summary)

# Create the main window
window = tk.Tk()
window.title("TripDoodler GPT Summarizer and Filter BOT")
window.configure(bg='#ADD8E6')  # Set the background color to blue

# Create the input box
input_label = tk.Label(window, text="Input Paragraph:")
input_label.pack()
input_box = tk.Text(window, height=10, width=50)
input_box.pack(fill=tk.BOTH, expand=True) # Allow the input box to expand

# Create the output box
output_label = tk.Label(window, text="Summary:")
output_label.pack()
output_box = tk.Text(window, height=5, width=50)
output_box.pack(fill=tk.BOTH, expand=True) #Allow the output box to expand

# Create the "Run" button
run_button = tk.Button(window, text="Run", command=run_summarizer, bg='#90EE90')
run_button.pack()

# Start the GUI event loop
window.mainloop()