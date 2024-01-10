#import packages
from flask import Flask, render_template, request, jsonify
import openai
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics.pairwise import cosine_similarity
# hide warnings
import warnings
warnings.filterwarnings('ignore')

#initiate Flask application
app = Flask(__name__)
#add API Key


#set up pretrained summarization model using Pegasus
model_name = "google/pegasus-xsum" #model created by Google
tokenizer = PegasusTokenizer.from_pretrained(model_name) #initiate tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu" #use GPU where available
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device) #load the model

def get_completion(prompt, model=model_name):
    # function to get reponses in the chat
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # creating deterministic model
    )
    return response.choices[0].message["content"]


# READ IN THE DATA
def create_idm(csv_path):
    df = pd.read_csv(csv_path)
    # drop all rows with NaN values for Keywords
    df = df.dropna(subset=['Keywords'])
    # drop all columns except for law_title, law_text, state, and keywords
    df = df[['law_title', 'law_text', 'location', 'Keywords']]
    keywords_for_mult_select = []
    # create list of keywords
    for k in df['Keywords']:
        if type(k) == float:
            continue
        k = k.split(',')
        k = [x.strip() for x in k]
        for l in k:
            l = l.strip()
            l = l.lower()
            keywords_for_mult_select.append(l)
    keywords_for_mult_select = list(set(keywords_for_mult_select))
    keywords_for_mult_select.sort()
    df = df[['location', 'law_title', 'law_text']]
    # add blank columns for each keyword
    for keyword in keywords_for_mult_select:
        df[keyword] = np.nan
    # fill in the blank columns with 1 if the keyword is in the law_text
    for index, row in df.iterrows():
        for keyword in keywords_for_mult_select:
            if keyword in row['law_text']:
                df.loc[index, keyword] = 1
            else:
                df.loc[index, keyword] = 0
    inverse_document_matrix = df

    # add a column for the index
    inverse_document_matrix['index'] = inverse_document_matrix.index

    # put the index column first
    cols = inverse_document_matrix.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    inverse_document_matrix = inverse_document_matrix[cols]
    return inverse_document_matrix


#function to make the string like the keywords: 
def clean_text(text):
    """
    Cleans the text by removing punct, stop words, and lemmatizing the words in the law 
    args: text
    returns: cleaned text (text)
    """
    #constants 
    punct = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" 
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = "".join([char for char in text if char not in punct])
    text = re.sub('\s+',' ',text) 
    #remove all numbers 
    text = re.sub("\d", ' ', text) 
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    text = " ".join(text)
    return text

def summarize(input_text):
    # function to summarizes text input from the chat
    input_text = "summarize: " + input_text # show the model what needs to be summarized
    # tokenize inputs and enable truncation
    tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    # generate summary (min and max can be adjusted)
    summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
    # decode summary into a string
    summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
    return summary


#function to search and rank the keywords and return a matrix
def find_most_similar_law(location, user_text, inverse_document_matrix):
    """
    Finds the most similar law to the user's text
    args: cleaned_user_text: the user's text that has been cleaned
          inverse_document_matrix: the inverse document matrix
    returns: the most similar law (law_title)
    """
    #clean the user's text
    cleaned_user_text = clean_text(user_text)
    #filter the inverse document matrix by the location
    filtered_idm = inverse_document_matrix[inverse_document_matrix['location'] == location]
    user_df = pd.DataFrame(columns=filtered_idm.columns)
    new_row = pd.Series(index=filtered_idm.columns) # adding new blank row (so user_df isn't blank)
    user_df = pd.concat([user_df, pd.DataFrame([new_row])], ignore_index=True) # officially adding it
    user_df[filtered_idm.columns[4:]] = 0 # setting all keywords to zero as default
    user_df['law_text'] = cleaned_user_text # setting the user text as the "law"
    user_df['law_title'] = 'user_text'
    user_df['location'] = location

    # fill in the blank columns with 1 if the keyword is in the law_text
    for index, row in user_df.iterrows():
        for keyword in filtered_idm.columns[4:]:
            if keyword in row['law_text']:
                user_df[keyword] = 1
            else:
                user_df[keyword] = 0
    

    dictionary_of_arrays = {}
    for l in filtered_idm['law_title']:
        dictionary_of_arrays[l] = filtered_idm[filtered_idm['law_title'] == l].iloc[:, 4:].values

    user_array = user_df.iloc[:, 4:].values

    dictionary_for_finding_similar_law = {}
    for law in dictionary_of_arrays:
        dictionary_for_finding_similar_law[law] = cosine_similarity(dictionary_of_arrays[law], user_array)[0][0]
    
    most_similar_law = max(dictionary_for_finding_similar_law, key=dictionary_for_finding_similar_law.get)

    return most_similar_law

@app.route("/")
def home():  
    # define path to home page
    return render_template("index.html")

@app.route("/get")
def get_bot_response():   
    # define path to model
    userText = request.args.get('msg') # get input
    response = get_completion(userText) # get response  
    #return str(bot.get_response(userText)) 
    return response


@app.route("/summarize", methods=["POST"])
def get_summary():
    # define path for text summarization
    try:
        input_text = request.form.get('text', '')  # get text from the input
        summary = summarize(input_text) # get summary of the text
        return jsonify({"summary": summary})
    except Exception as e:
        print(e)  # log the exception for debugging
        return jsonify({"error": "An error occurred during summarization."}), 500

@app.route("/process_input", methods=["POST"])
def process_input():
    try:
        location = request.form.get('input1', '')
        user_issue = request.form.get('input2', '')
        cleaned_text = clean_text(user_issue)
        summary = summarize(cleaned_text)

        # Assuming create_idm returns the dataframe required for find_most_similar_law
        idm = create_idm("../data/clean_data.csv")
        most_similar_law = find_most_similar_law(location, cleaned_text, idm)

        return jsonify({"most_similar_law": most_similar_law, "summary": summary})
    except Exception as e:
        print(e)  # log the exception for debugging
        return jsonify({"error": "An error occurred processing the inputs."}), 500


# run the script above
if __name__ == "__main__":
    app.run()