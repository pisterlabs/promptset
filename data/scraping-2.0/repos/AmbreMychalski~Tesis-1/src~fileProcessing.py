import os
import requests
import json
import openai
import re
import PyPDF2
import pandas as pd
import tiktoken

openai.api_key = ""
openai.api_base = "https://invuniandesai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

max_tokens = 500

rawDataset = "front/rawDataset/"
txt_directory = "front/ProcessedDataset/txt/"
scraped_directory = "front/ProcessedDataset/scraped/"
embeddings_directory = "front/embeddings/"

def process_to_txt():

    for filename in os.listdir(rawDataset):
        # Ouvrir le fichier PDF en mode lecture binaire ('rb')
        with open(rawDataset+filename, 'rb') as pdf_file:
            # Créer un objet PDFReader
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Initialiser une variable pour stocker le texte brut
            raw_text = ''

            # Parcourir chaque page du PDF
            for page_num in range(len(pdf_reader.pages)):
                # Extraire le texte de la page
                page = pdf_reader.pages[page_num]
                raw_text += "###" + str(page_num+1)+ "###" + page.extract_text() + "\n"

        raw_text = raw_text.replace(",",' ')
        filename=filename.replace(".pdf",".txt")
        # Si vous souhaitez sauvegarder le texte brut dans un fichier texte
        with open(txt_directory+filename, 'w', encoding='utf-8') as txt_file:
            txt_file.write(raw_text)

def remove_newlines(serie):
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def txt_to_scraped():

    # Create a list to store the text files
    texts=[]
    # Get all the text files in the text directory
    for file in os.listdir(txt_directory):
        # Open the file and read the text
        with open(txt_directory + file, "r", encoding="UTF-8") as f:
            text = f.read()
            texts.append([file,text])

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])
    print(df['fname'])
    # Set the text column to be the raw text with the newlines removed
    df['text'] = remove_newlines(df.text)
    df.to_csv(scraped_directory+'scraped.csv')

def split_into_many(text, tokenizer, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('\n')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []
    last_page_number = '0'
    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        page_number = re.findall(r'###(\d+)###', sentence)
        sentence = re.sub(r'###\d+###', '', sentence)
        # print(sentence)
        if len(page_number)>0:
            last_page_number=page_number[0]
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
        chunk.append('###'+last_page_number+"###"+sentence)
        tokens_so_far += token + 1

    return chunks

def scraped_shortened():
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv(scraped_directory+'/scraped.csv', index_col=0)

    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    shortened = []
    # Loop through the dataframe
    for row in df.iterrows():
        temp=[]
        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            temp += split_into_many(text=row[1]['text'], tokenizer=tokenizer)
            for text in temp:
                data=[]
                #print(text)
                page_nb = re.findall(r'###(\d+)###', text)
                page_nb=list(set(page_nb))
                text = re.sub(r'###\d+###', '', text)
                # print(page_nb, text)
                data.append(row[1]['title'])
                data.append(page_nb)
                data.append(text)
                shortened.append(data)

        # Otherwise, add the text to the list of shortened texts
        else:
            for text in temp:
                data=[]
                page_nb = re.findall(r'###(\d+)###', text)
                page_nb=list(set(page_nb))
                text = re.sub(r'###\d+###', '', text)
                data.append(row[1]['title'])
                data.append(page_nb)
                data.append(row[1]['text'] )
                shortened.append(data)

    df = pd.DataFrame(shortened, columns = ['title','page_number','text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return(df)

def df_to_embed(df):
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine="text-embedding-ada-002-rfmanrique")['data'][0]['embedding'])
    df.to_csv(embeddings_directory+'embeddings.csv')

if __name__ == '__main__':
    process_to_txt()
    print('----process to text complete----')
    txt_to_scraped()
    print('----text to scraped complete----')
    df = scraped_shortened()
    print('----scraped shortened complete----')
    df_to_embed(df)
    print('----df to embeddings complete----')

