import fitz
import tkinter as tk
from tkinter import filedialog
import webbrowser
import nltk
from threading import Thread
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from summarizer import Summarizer
import openai 

openai.api_key = 'OPENAI_API_KEY' # Set OpenAI API key

nltk.download('punkt')
nltk.download('stopwords')


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text


def get_word_meaning(word):
    lemmatizer = WordNetLemmatizer()
    lemmatized_word = lemmatizer.lemmatize(word)

    synsets = wordnet.synsets(lemmatized_word)
    meanings = []
    for synset in synsets:
        meanings.extend(lemma.name() for lemma in synset.lemmas())

    return list(set(meanings))


def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
    return file_path


def process_pdf_and_get_meanings(pdf_path, words_to_lookup, result_text):
    for word in words_to_lookup:
        meanings = get_word_meaning(word)
        if meanings:
            result_text.insert(tk.END, f"Meaning of '{word}': {', '.join(meanings)}\n")
        else:
            result_text.insert(tk.END, f"No meaning found for '{word}'\n")


def generate_summary(pdf_path, result_text):
    try:
        pdf_text = extract_text_from_pdf(pdf_path)
        summary = generate_chatgpt_summary(pdf_text)
        result_text.insert(tk.END, f"\nGenerated Summary:\n{summary}\n")
    except Exception as e:
        result_text.insert(tk.END, f"Error generating summary: {str(e)}\n")


def generate_chatgpt_summary(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Change for different engines
        prompt=text,
        max_tokens=150, 
        temperature=0.7,  
        stop=None 
    )
    summary = response['choices'][0]['text']
    return summary

def on_open_pdf(pdf_path_var, pdf_viewer, result_text):
    pdf_path = pdf_path_var.get()
    if pdf_path:
        display_pdf(pdf_path, pdf_viewer)
        result_text.delete(1.0, tk.END)  # Clear previous results
        result_text.insert(tk.END, "PDF displayed. Enter words for meaning or generate summary:\n")
    else:
        result_text.insert(tk.END, "No PDF file selected.\n")

def display_pdf(pdf_path, pdf_viewer):
    doc = fitz.open(pdf_path)
    pdf_text = ""

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pdf_text += page.get_text()

    doc.close()

    pdf_viewer.delete(1.0, tk.END)
    pdf_viewer.insert(tk.END, pdf_text)

def on_submit(pdf_path_var, words_var, result_text):
    pdf_path = pdf_path_var.get()
    words_to_lookup = [word.strip() for word in words_var.get().split(",")]

    if pdf_path:
        Thread(target=process_pdf_and_get_meanings, args=(pdf_path, words_to_lookup, result_text)).start()
    else:
        result_text.insert(tk.END, "No PDF file selected.\n")

def on_generate_summary(pdf_path_var, result_text):
    pdf_path = pdf_path_var.get()
    if pdf_path:
        Thread(target=generate_summary, args=(pdf_path, result_text)).start()
    else:
        result_text.insert(tk.END, "No PDF file selected.\n")

def get_selected_word_meanings(event, pdf_viewer, result_text):
    selected_text = pdf_viewer.get(tk.SEL_FIRST, tk.SEL_LAST)
    selected_words = [word.strip() for word in selected_text.split()]
    
    if selected_words:
        meanings = [get_word_meaning(word) for word in selected_words]
        for word, meaning in zip(selected_words, meanings):
            result_text.insert(tk.END, f"Meaning of '{word}': {', '.join(meaning)}\n")
            
        # Create a popup window to display meanings directly above the selected text
        popup_window = tk.Toplevel()
        popup_text = tk.Text(popup_window, wrap=tk.WORD, height=len(selected_words), width=30)
        popup_text.config(state=tk.NORMAL) 
        popup_text.insert(tk.END, "\n".join([f"{word}: {', '.join(meaning)}" for word, meaning in zip(selected_words, meanings)]))
        popup_text.grid(row=0, column=0, padx=5, pady=5)
    else:
        result_text.insert(tk.END, "No words selected.\n")

def main():
    root = tk.Tk()
    root.title("PDF Reader and Word Extractor")

    pdf_path_label = tk.Label(root, text="Select PDF File:")
    pdf_path_var = tk.StringVar()
    pdf_path_entry = tk.Entry(root, textvariable=pdf_path_var, state="disabled")
    pdf_path_button = tk.Button(root, text="Browse", command=lambda: pdf_path_var.set(open_file_dialog()))

    open_pdf_button = tk.Button(root, text="Open PDF", command=lambda: on_open_pdf(pdf_path_var, pdf_viewer, result_text))
    generate_summary_button = tk.Button(root, text="Generate Summary", command=lambda: on_generate_summary(pdf_path_var, result_text))

    words_label = tk.Label(root, text="Enter Words (comma-separated):")
    words_var = tk.StringVar()
    words_entry = tk.Entry(root, textvariable=words_var)

    submit_button = tk.Button(root, text="Submit", command=lambda: on_submit(pdf_path_var, words_var, result_text))

    pdf_viewer = tk.Text(root, wrap=tk.WORD, height=20, width=80)
    pdf_viewer.config(state=tk.NORMAL) 

    result_text = tk.Text(root, wrap=tk.WORD, height=10, width=50)
    result_text.config(state=tk.NORMAL) 

    get_selected_word_meanings_button = tk.Button(root, text="Get Selected Word Meanings")

    # Bind the event to get selected word meanings instantly
    pdf_viewer.bind("<ButtonRelease-1>", lambda event: get_selected_word_meanings(event, pdf_viewer, result_text))

    pdf_path_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    pdf_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
    pdf_path_button.grid(row=0, column=2, padx=5, pady=5)

    open_pdf_button.grid(row=1, column=0, pady=10)
    generate_summary_button.grid(row=1, column=1, pady=10)

    words_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    words_entry.grid(row=2, column=1, padx=5, pady=5, sticky="we")

    submit_button.grid(row=3, column=0, columnspan=3, pady=10)

    pdf_viewer.grid(row=4, column=0, columnspan=3, padx=5, pady=5)
    result_text.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
    get_selected_word_meanings_button.grid(row=6, column=0, columnspan=3, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()