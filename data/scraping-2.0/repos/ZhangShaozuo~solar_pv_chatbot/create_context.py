import requests
import openai
from openai import distances_from_embeddings
import numpy as np
import os
import pandas as pd
import tiktoken
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import time
from transformers import pipeline
import torch
from torch import cosine_similarity

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def pdf_extractor(path_src, path_tgt = 'processed/scraped.csv'):
    # creating a pdf reader object
    text_store = []
    for pdf in os.listdir(path_src):
        if pdf.endswith(".pdf"):
            pdf_path = os.path.join(path_src, pdf)
            reader = PdfReader(pdf_path)
            text_substore = ''
            txt_path = pdf_path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:            
                for page in reader.pages:
                    text = page.extract_text().replace('\t',' ')
                    text_substore = text_substore + text
                    f.write(text)
            text_store.append((txt_path, text_substore))
    df = pd.DataFrame(text_store, columns = ['fname', 'text'])
    df['text'] = remove_newlines(df.text)
    df.to_csv(path_tgt)

def split_into_many(text, tokenizer, max_tokens):
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
    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

def create_embed(path, tokenizer, max_tokens = 256):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df = pd.read_csv(path, index_col=0)
    df.columns = ['title', 'text']
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    fig, ax = plt.subplots()
    df.hist('n_tokens', ax=ax)
    fig.savefig('processed/n_tokens.png')
    shortened = []
    for row in df.iterrows():
        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue
        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'], tokenizer, max_tokens)  
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x))) 
    fig, ax = plt.subplots()
    df.hist('n_tokens', ax=ax, bins=10)
    fig.savefig('processed/n_tokens_split.png')
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv(f'processed/embeddings_{max_tokens}.csv')

def create_context(
    question, df, max_len=256, size = 'ada'
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
    return q_embeddings, "\n\n###\n\n".join(returns)

def prompt(context, question):
    return f"Answer the question below, you can refer to but NOT limited to the contexts \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"

def answer_question(
    df,
    data_df,
    model : str,
    max_len : int,
    debug: bool,
    max_tokens : int,
    embed_size : int,
    size="ada",
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pipeline(model="declare-lab/flan-alpaca-gpt4-xl", device=2)
    contexts, responses, cos_sims =[], [], []
    for row in data_df.iterrows():
        q_embeddings, context = create_context(
            row[1]['content'],
            df,
            max_len=max_len,
            size=size,
        )
        
        a_embeddings = openai.Embedding.create(input = row[1]['reply'], engine='text-embedding-ada-002')['data'][0]['embedding']
        if debug:
            print("Context:\n" + context+'\n\n')
        if model == "text-davinci-003":
            proc = False
            while not proc:
                try:
                    response = openai.Completion.create(
                        prompt = prompt(context, row[1]['content']),
                        temperature = 0,
                        max_tokens = max_tokens,
                        top_p = 1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=stop_sequence,
                        model=model,
                    )["choices"][0]["text"].strip()
                except Exception as e:
                    print(e)
                    time.sleep(5)
                else:
                    proc = True
        # elif model == 'gpt-3.5-turbo' or model == 'gpt-4':
        #     response = openai.ChatCompletion.create(
        #         model=model,
        #         messages=[
        #             {"role": "user", "content": prompt(context, row[1]['content'])},
        #         ]
        #     )["choices"][0]["message"]['content'].strip()
            
        response = model(prompt(context, row[1]['content']), max_length=512, do_sample=True)[0]['generated_text']
        contexts.append(context)
        responses.append(response)
        cos_sims.append(cosine_similarity(q_embeddings, a_embeddings))

    data_df['contexts'] = contexts    
    data_df['preds'] = responses
    data_df['cos_sims'] = cos_sims
    # data_df.to_csv(f'data_{model}_{max_len}c.csv', index=False)
    data_df.to_csv(f'data_flan_alpaca_{embed_size}e_{max_len}c.csv', index=False)

def crawl(source):
    name = 'ema'
    os.mkdir("text/", exist_ok=True)
    os.mkdir("text/" + name + "/", exist_ok=True)
    os.mkdir("processed", exist_ok=True)
    # While the queue is not empty, continue crawling
    for url, ref_name in source.items():
        fp = os.path.join('text', name, ref_name)
        with open(fp, "w", encoding="UTF-8") as f:
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            text = soup.get_text()
            f.write(text)

def main_srcs():
    # full_url = {"https://www.ema.gov.sg/Guide_to_Solar_PV.aspx": 'guide_to_solar_pv.txt',
    # }
    # crawl(full_url)
    path_head = 'text/ema_src'
    pdf_extractor(path_head)

def main_embeds():
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    create_embed('processed/scraped.csv', tokenizer, max_tokens = 64)

def main_qa():
    
    for embed_size in [64,128,256]:
        df=pd.read_csv('processed/embeddings_128.csv', index_col=0)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

        # data_df = pd.read_csv('data_text-davinci-003.csv')
        data_df = pd.read_csv('data_w_pred.csv')[['content','reply']]
        # model = "text-davinci-003"
        model = 'Flan-GPT4All-XL'
        for max_len in [256,512,768]:
            answer_question(df, data_df, 
                            model=model, 
                            max_len = max_len,
                            debug = True,
                            max_tokens = 128,
                            embed_size = embed_size,
                            )
def eval_qa():
    es = [64,128,256]
    cs = [256,512,768]
    # model = "text-davinci-003"
    model = 'flan_alpaca'
    for e in es:
        for c in cs:
            data_df = pd.read_csv(f'data_{model}_{e}e_{c}c.csv')
            print('Embedding size: ', e, 'Context size: ', c, 'Accuracy: ', data_df['cos_sims'].mean())
if __name__ == '__main__':
    # main_srcs()
    # main_embeds()
    # main_qa()
    eval_qa()