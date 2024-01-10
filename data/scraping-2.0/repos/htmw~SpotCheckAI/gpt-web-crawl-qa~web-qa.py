#import dependencies
import pandas as pd
import tiktoken
import openai

##########
##Step 1##
##########

#manual data scrape
data = [
    ['spotcheckai.web.app home SpotCheckAI Home', "SpotCheckAI Skin Cancer Detector Using AI AI-powered skin cancer detection is a promising field that can help improve the accuracy and speed of identifying different types of skin lesions, including melanoma. By training machine learning algorithms on large datasets of skin images, researchers and healthcare professionals can develop tools to enable earlier intervention and successful treatment. Check It Out Feedback We want to know what you think of our platform! Please leave your feedback in the Contact Us page!"],
    ['spotcheckai.web.app about SpotCheckAI About Us', "Meet the Developer Rafferty Leung Rafferty Leung is a Master of Computer Science student at Pace University. He is interested in the intersection between computer science and life sciences. For more information about him please visit: rafferty leung.com github.com htmw SpotCheckAI Feedback We want to know what you think of our platform! Please leave your feedback in the Contact Us page!"],
    ['spotcheckai.web.app info SpotCheckAI Additional Information', "Melanoma Melanoma is a type of skin cancer that develops in the pigment-producing cells of the skin, known as melanocytes. It is the most dangerous form of skin cancer and can be life-threatening if not treated early. Melanoma is often caused by exposure to ultraviolet (UV) radiation from the sun or tanning beds, although other factors such as family history, fair skin, and a weakened immune system can also increase the risk of developing this disease. Benign Melanoma Benign melanoma, also known as a nevus or mole, is a non-cancerous form of the disease that usually does not pose any health risks. These moles are typically small, round, and evenly colored, and may appear anywhere on the body. While benign melanomas are generally harmless, it is important to monitor them for changes in size, shape, or color, as well as the development of symptoms such as itching or bleeding. In some cases, benign melanomas may need to be removed for cosmetic or medical reasons Cancerous Melanoma Cancerous melanoma is a malignant form of the disease that can spread to other parts of the body, including the lymph nodes, lungs, liver, and brain. It is characterized by the abnormal growth and proliferation of melanocytes, which can result in the formation of dark, irregularly shaped moles or lesions on the skin. Symptoms of cancerous melanoma can include changes in the size, shape, or color of existing moles, the appearance of new moles, and the development of sores that do not heal. Feedback We want to know what you think of our platform! Please leave your feedback in the Contact Us page!"],
    ['spotcheckai.web.app form SpotCheckAI Prediction Form', "Disclaimer Important Message By utilizing this tool, you agree that this is not a replacement for a physician and that you should always consult a physician for any medical concerns. Instructions 1. Click the upload button. 2. Select the photo of interest. 3. Click the submit button. 4. The page will send the data to the model and will output a result. In Development: Uploading Directly From Camera Current Model Metrics Model ID: ImageClassifier02222023 Loss: 0.3981 Accuracy: 0.815625 Precision: 0.74556214 Recall: 0.8873239 Submit Photo Only Image Files are Accepted png, jpeg, jpg, bmp No file chosen Results If result is less than 0.5, the model predicts that the lesion is benign non-cancerous. If the result is greater than 0.5, the model predicts that the lesion is cancerous. If the predicted result is showing an error, please see contact us for further assistance. Predicted Result:"],
    ['spotcheckai.web.app chat SpotCheckAI Chat Bot', "Chat with our GPT powered bot if you have any questions about your results or the website in general."],
    ['spotcheckai.web.app contact SpotCheckAI Contact', "Contact us if you have any questions or concerns."]
]

##########
##Step 2##
##########

#clean up data array
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

##########
##Step 3##
##########

# Create a dataframe from the list of texts
df = pd.DataFrame(data, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

##########
##Step 4##
##########

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

##########
##Step 5##
##########

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )

##########
##Step 6##
##########

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()

##########
##Step 7##
##########

openai.api_key = <YOUR-API-KEY-HERE>
df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()

##########
##Step 8##
##########

from openai.embeddings_utils import distances_from_embeddings

df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

df.to_csv('processed/embeddings.csv')
df.head()

##########
##Step 9##
##########

import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()

#############
###Step 10###
#############

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know. Please email us so that we can further assist you\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

#############
###Step 11###
#############

answer_question(df, question="My result is 0.8, what does this mean?", debug=False)