import torch
from torch.utils.data import Dataset
import sklearn
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer
from flask import Flask, render_template, request

##import datasets and merge them

import pandas as pd

import openai
import os

app = Flask(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]

#*
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Started!")
#print functions print the column names of each DataFrame. You can then compare the column names used in your merge operations with the actual column names in your CSV files. If there is a mismatch, you may need to update the merge operations accordingly.
questions = pd.read_csv('Questions.csv', encoding='latin1').head(5)
#print(f"questions.columns: {questions.columns}")
answers = pd.read_csv('Answers.csv', encoding='latin1').head(5)
#print(f"answers.columns: {answers.columns}")
tags = pd.read_csv('Tags.csv', encoding='latin1').head(5)
#print(f"tags.columns: {tags.columns}")

# Merge questions and answers
qa = pd.merge(questions, answers, left_on='Id', right_on='ParentId', suffixes=('_question', '_answer'))

# check the column names by printing the DataFrame column names
#print(f"qa.columns: {qa.columns}")
#print(f"tags.columns: {tags.columns}")

# convert the 'Id' column data type to integer.
tags['Id'] = tags['Id'].astype(int)

# Merge tags
data = pd.merge(qa, tags, left_on='Id_question', right_on='Id', suffixes=('', '_tag'))

#print(f"data.columns: {data.columns}")
    
# You can use the train_test_split function from the sklearn library to split your data into training and testing sets. Here's an example of how to do this:
# In this example, we're splitting the data into training and testing sets, with 80% of the data used for training and 20% used for testing. The random_state parameter ensures that the split is reproducible.
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#Finally, you can split the training data into training and validation sets. This step is similar to step 3, but it ensures that you have a separate set of data to validate your model during the training process. Here's an example of how to split the training data into training and validation sets:
#In this example, we're splitting the training data into training and validation sets, with 90% of the data used for training and 10% used for validation.
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

#That's it! You've now split your dataset into training, validation, and testing sets in Python using PyTorch.

##get started with GPT-2
# Load the pre-trained GPT-2 model from the Hugging Face Transformers library

#from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained('gpt2')
#model = GPT2Model.from_pretrained('gpt2')

#def recursive_tokenizer(text: str) -> str:
#    try:
#        return tokenizer.encode(text)
#    except Exception as ex:
#        logging.warning("Sequence length too large for model, cutting text in half and calling again")
#        return recursive_tokenizer(text=text[:(len(text) // 2)]) + recursive_tokenizer(text=text[(len(text) // 2):],)

#def split_text_into_chunks(text, max_tokens=1024):
#    tokens = tokenizer.encode(text)
#    token_chunks = []
#
#    current_chunk = []
#    current_chunk_len = 0

#    for token in tokens:
#        token_len = len(tokenizer.decode([token]))

#        if current_chunk_len + token_len > max_tokens:
#            token_chunks.append(current_chunk)
#            current_chunk = []
#            current_chunk_len = 0

#        current_chunk.append(token)
#        current_chunk_len += token_len

#    if current_chunk:
#        token_chunks.append(current_chunk)

#    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

#    return text_chunks

#input_text = "Your long input text goes here..."
#text_chunks = split_text_into_chunks(input_text)

#for chunk in text_chunks:
    # Process the chunk with your model, e.g., generate responses, summarize, etc.
#    pass

#***

#def truncate_text(text, max_tokens=1024):
#    tokens = tokenizer.encode(text)
#    truncated_tokens = tokens[:max_tokens]
#    truncated_text = tokenizer.decode(truncated_tokens)
#    return truncated_text

#input_text = "Your long input text goes here..."
#truncated_text = truncate_text(input_text, max_tokens=1024)

# Process the truncated_text with your model, e.g., generate responses, summarize, etc.

#***

# Load the pre-trained GPT-3 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
model = AutoModelForCausalLM.from_pretrained("openai-gpt")

# Prepare the optimizer
optimizer = AdamW(model.parameters())

# Prepare your training data
# This can be a dataset of conversation transcripts
# that you want the model to learn from
# You can use the tokenizer to encode the text data

print("Started 2!")
#print functions print the column names of each DataFrame. You can then compare the column names used in your merge operations with the actual column names in your CSV files. If there is a mismatch, you may need to update the merge operations accordingly.
questions = pd.read_csv('Questions.csv', encoding='latin1').head(5)
#print(f"questions.columns: {questions.columns}")
answers = pd.read_csv('Answers.csv', encoding='latin1').head(5)
#print(f"answers.columns: {answers.columns}")
tags = pd.read_csv('Tags.csv', encoding='latin1').head(5)
#print(f"tags.columns: {tags.columns}")

# Merge questions and answers
qa = pd.merge(questions, answers, left_on='Id', right_on='ParentId', suffixes=('_question', '_answer'))

# check the column names by printing the DataFrame column names
#print(f"qa.columns: {qa.columns}")
#print(f"tags.columns: {tags.columns}")

# convert the 'Id' column data type to integer.
tags['Id'] = tags['Id'].astype(int)

# Merge tags
data = pd.merge(qa, tags, left_on='Id_question', right_on='Id', suffixes=('', '_tag'))

#print(f"data.columns: {data.columns}")

encoded_data = [tokenizer.encode(text, truncation=True) for text in data.Body_question]

#define the num_training_steps variable before using it in your training loop
#define num_training_steps to be a value that is less than or equal to the length of the encoded_data list. You can use the len() function to get the length of the encoded_data list, like this:
num_training_steps = len(encoded_data)

#Then you can use this value to iterate over the encoded_data list:
# Fine-tune the model on the training data
# train the model for 10 epochs
for epoch in range(10):
    print(f"Epoch {epoch}")
    for i in range(num_training_steps):
        optimizer.zero_grad()
        input_ids = torch.tensor(encoded_data[i]).unsqueeze(0)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

# Use the fine-tuned model for your NLP chat application
# You can use the `model.generate` method to generate responses
# based on the input provided by the user

def generate_response(prompt, max_tokens=1024, temperature=0.7):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=max_tokens,
      n=1,
      stop=None,
      temperature=temperature
    )
    return response["choices"][0]["text"]

try:
    prompt = "How do I learn Python?"
    response = generate_response(prompt, max_tokens=100)
    print(response)
except Exception as e:
    print(f"Error: {e}")
    print("I'm sorry, there was an error processing your request.")

#app

# Define your Flask route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    # Get the user's question from the form
    question = request.form['question']

    # Call your AI model to generate an answer based on the user's question
    response = generate_response(question)

    # Return the AI model's response to the user
    return render_template('index.html', answer=response)

