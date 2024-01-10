import pandas as pd
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import datetime
import os
import csv

from config import CONFIG

# Read the embeddings
#df=pd.read_csv(CONFIG['embeddings_csv'], index_col=0)
df=pd.read_csv(CONFIG['embeddings_csv'])
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
#print(df.head())



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
    titles = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        #print(row)
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row['text'])
        titles.append(row['title'])
    # Return the context
    #print(titles)
    return "\n\n###\n\n".join(returns)


def append_to_csv(filename, row_dict):
    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    # If the file doesn't exist, create it with the column names
    fieldnames = row_dict.keys()
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Write the new row to the CSV file
        writer.writerow(row_dict)

def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="What can you tell me about Juan Merlos?",
    max_len=1800,
    size="ada",
    debug=True,
    max_tokens=150,
    stop_sequence=None,
    language="English",
    log_answer=False,
    answers_log_file='./logs/answers.log'
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
        debug=false,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant that replies with concrete answers based on the context provided by the user. If you dont know the answer, you admit it and invite the user to pose it in another way"},
            {"role": "user", "content": f"Based on the following conte\n\nxt: {context}\n\n answer the following question: \n\n {question}\n"},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=conversation,
            max_tokens=max_tokens,
            stop=stop_sequence,
            temperature=0,
            n=1,
            
        )
        #print(response)
        answer = response['choices'][0]['message']['content']


        
        if log_answer:
            try:
                row = {
                    'date': datetime.date.today(),
                    'question': question,
                    'system': conversation[0],
                    'prompt': conversation[1],
                    'answer': answer
                }
                append_to_csv(answers_log_file, row)
            except Exception as e:
                print(e)
                
        return answer
    
    except Exception as e:
        print(e)
        return "There was an issue while processing your question... :("