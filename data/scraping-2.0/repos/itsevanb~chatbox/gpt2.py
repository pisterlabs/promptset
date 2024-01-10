import torch
import tkinter as tk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import openai 

#lodaing GPT-2 and tokenizer 
model_name = 'ChatGPT4 Evbot'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
sentiment_analysis = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

#create a function to generate respones
def generate_response(input_text, max_length=100, num_return_sequences=1): 
#max_length - 100 tokens for generated response, r_n_s - model should only generate one sequence 
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
#tokenizes input_text, output should be a pytorch tensor, a pyt-tensor is a data structure similar #to an array or vector -- a container for numerical data that you can perform mathematical #operations on, utilizes GPU instead of CPU
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
#creates another pytorch sensor that is same shape as other tensor called 'input_ids'/ 'input_ids' #is a sequence of tokens(words) but represented as integers(whole numbers) and each integer #correspnds to said token/atenntion_mask will be used to mask out tokens that are not relevant
#(wont speak gibberish)
    pad_token_id = tokenizer.eos_token_id
#EOS token is specific token that is used to indicate the current sentence or sequence has ended and #to generate a new one/prevents chatbot from treating padding tokoens as part of the input sequence/#EOS token used to indicate end of sentence or sequence, sets pad_token_id = EOS token
    output_sequence = model.generate( 
#calls generate method on GPT-2 model and takes number of paramerters
        input_ids = input_ids,
        max_length = max_length,
        num_return_sequences = num_return_sequences,
        no_repeat_ngram_size = 2, 
#size of ngrams to avoid repeating in generated text/helps prevent repetitve phrases
#n-grams are contiguous(words or characters that appear one after the other) sequence of n items #from a sample of text or speech
        do_sample = True,
        temperature = 0.7,
#controls randomness of generated text, higher values the more random lower values more deterministic
        top_k = 50,
#reduces probablity of generating low probability tokens
        top_p = 0.95,
#uses nucleus sampling(more creative and natural sounding output)(picks next token based on #probability threshold rather than always picking most likely token)threshold or nucleus is the #probability value that determines how many possible next words to consider, rather than looking at #most probably tokens(top_k sampling\\top_p is subset of top_k)
        attention_mask = attention_mask,
        pad_token_id = pad_token_id,
    )
    decoded_output = [tokenizer.decode(sequence) for sequence in output_sequence]
#decodes generated token sequences back into readible text iterating through out_putsequences
    return decoded_output

#GUI for chatbot
def on_send():
    user_input = user_input_var.get()
#gets users input from entry widget using get() method 
    chatbot_response = generate_response(user_input)[0]
#generates chatbots response ny calling generate response function
    sentiment = sentiment_analysis(user_input)
    max_length = max(2, len(user_input.split()) // 2)
    max_length = max(max_length, 25)
    summary = summarizer(user_input, max_length = max_length, min_length = 25)
#uses 'Hugging Face' pipelines(open-source library for building and training natural langauge #proscessing)
    chatbox.config(state=tk.NORMAL)
    chatbox.insert(tk.END, f"User: {user_input}\n")
    chatbox.insert(tk.END, f"Evbot: {chatbot_response}\n")
    chatbox.insert(tk.END, f"{sentiment[0]['label']} ({sentiment[0]['score']:.2f})\n")
    chatbox.insert(tk.END, f"Summary: {summary[0]}['summary_text']\n\n")
#lines insert user input, chatbot response, sentiment analysis, and summarization results into text
#widget 'tk.end' tells 'insert()' to append the text at end of widget
    user_input_var.set("")
#clears entry widget 

root = tk.Tk()
#creates main tinker window called the root window
root.title("Evbot")
#title of root window
user_input_var = tk.StringVar()
#creates object used to store and retrieve the users input from widget
chatbox = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED)
chatbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
#creates text widget 'chatbox' wraps the word,pack is called to add widget to root window with #10pixel padding
input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=(0, 10), fill = tk.X)
#creates a frame widget, containing the entry widget and 'send' button, pack is called to add frame #to root window with padding on the left, right, and buttom sides
user_input_entry = tk.Entry(input_frame, textvariable=user_input_var)
user_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
#creates a entry widget to get users input, pack is called to add widget to root window
send_button = tk.Button(input_frame, text="Send", command=on_send)
send_button.pack(side=tk.RIGHT)

root.mainloop()


if __name__ == '__main__':
    print(f'\033[92m''Welcome to ChatBox :)\nTo exit please enter "quit" ''\033[0m')

    while True: 
#creates an infinite loop unless user input 'quit'
        user_input = input('\033[91m'"User:") 
        if user_input.lower() == 'quit':
            break
        chatbot_response = generate_response(user_input)[0]
#calls generate response function and passes 'user_input' as an argument since generate response #returns a list of responses the '[0]' only returns the first reposnse from the list
        print(f"'\033[94m'Evbot: {chatbot_response}'\033[0m'")