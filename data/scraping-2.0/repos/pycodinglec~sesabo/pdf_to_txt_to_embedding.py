"""
PDF 파일들로부터 텍스트 파일을 추출하고, 
documents_txt에 개별저장 후 파일 별 임베딩 생성
"""
from pypdf import PdfReader
import os
import openai
import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import pickle
import pandas as pd

def pdf_to_txt():
    for pdffile in os.listdir('documents/now'):
        text = ''
        print('documents/now/'+pdffile) # for debug
        reader = PdfReader('documents/now/'+pdffile)
        for i in range(len(reader.pages)):
            text += reader.pages[i].extract_text()
        txtfilewithpath = 'documents_txt/'+pdffile.rstrip('.pdf')+'.txt'
        with open(txtfilewithpath, 'w', encoding='utf-8') as f:
            f.write(text)

def txt_to_embedding():
    dotenv.load_dotenv()
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    EMBEDDING_MODEL = "text-embedding-ada-002"
    GPT_MODEL = "gpt-3.5-turbo"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
    )
    enc = tiktoken.encoding_for_model(GPT_MODEL)

    for txtfile in os.listdir('documents_txt'):
        df = pd.DataFrame(columns=['text', 'embedding'])
        count = 0
        with open('documents_txt/'+txtfile, 'r') as f:
            original = f.read()
        splitted_text = text_splitter.split_text(original)
        for fragment in splitted_text:
            count += len(enc.encode(fragment))
        print(f'Token count of {txtfile} is {count},\nso ${count/1000*0.0001} will be charged.')
        #input('Press enter to continue.')
        length_of_splitted_text = len(splitted_text)
        for i in range(length_of_splitted_text):
            response = openai.Embedding.create(
                input = splitted_text[i],
                model = EMBEDDING_MODEL
            )
            df.loc[len(df)] = [splitted_text[i], response['data'][0]['embedding']]
            print(f'{i}/{length_of_splitted_text}') # for debug
        df.to_csv('documents_embed/'+txtfile.rstrip('.txt')+'.embed.csv', index = False)
        # with open('documents_embed/'+txtfile.rstrip('.txt')+'.embed.txt', 'w', encoding = 'utf-8') as f:
        #     f.write(str(embedding))
        with open('documents_embed/'+txtfile.rstrip('.txt')+'.embed.pkl', 'wb') as f:
            pickle.dump(df, f)

if __name__ == '__main__':
    # pdf_to_txt()
    txt_to_embedding()