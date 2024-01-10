import pandas as pd
from scraper import Scraper
import os

import contractions
import re
import nltk
nltk.download('stopwords')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import random

import openai

class Reccom:
    def __init__(self):
        self.data = None
        self.pure_data = None
        self.similar_products = None
    
    def load_data(self, rescraping=False):
        csv_name = '20_tokopedia_products.csv'

        if csv_name in os.listdir():
            data = pd.read_csv(csv_name)
        elif csv_name not in os.listdir() or rescraping:
            a = Scraper()
            data = a.get_data()
            a.driver.quit()

            data = pd.DataFrame(data)
            data.to_csv(csv_name, index=False)

        data['id'] = data.index
        pure_data = data.copy()
        
        self.data = data
        self.pure_data = pure_data
        
    def similarity(self, prod_id):
        # Need to be changed to experimental fill dataframe form later
        data = self.data.copy()
        pure_data = self.pure_data.copy()
        
        pure_data['category'] = ['wedding', 'wedding', 'baby', 'baby', 'wedding', 'general', 'baby', 'baby', 'baby', 'general', 'general', 'general', 'general', 'general', 'wedding', 'baby', 'wedding', 'general', 'general', 'wedding']
        data = pure_data.copy()
        
        # Cleaning the texts
        def txtprocess(txt):
            # Lower the texts
            txt = str(txt).lower()
            # Remove contractions
            txt = contractions.fix(txt)
            # Just pick the alphabet
            txt = re.sub(r'[^a-zA-Z]', ' ', txt)
            # Fix unnecessary space
            txt = re.sub(' +', ' ', txt)

            txt = ' '.join(txt.split())
            return txt

        data.Product = data.Product.map(txtprocess)
        data.Description = data.Description.map(txtprocess)
        
        # Cleaning stopwords
        stop_words = set(nltk.corpus.stopwords.words('indonesian'))
        stop_words.add('gift')
        stop_words.add('hampers')
        stop_words.add('hadiah')
        stop_words.add('kado')
        stop_words.add('x')

        def remove_stopwords(txt):
            no_stopword_txt = [w for w in txt.split() if not w in stop_words]
            return ' '.join(no_stopword_txt)

        data.Product = data.Product.map(remove_stopwords)
        data.Description = data.Description.map(remove_stopwords)
        
        # Feature Engineering
        data['char_count'] = data['Description'].map(len)
        data.Price = data.Price.str[2:].str.replace('.', '').astype(int)
        
        def get_similarity_matrix(
            weight_product = 0.4,
            weight_description = 0.3,
            weight_prices = 0.2,
            weight_char_count = 0.1
        ):

            # For product and description
            tfidf_product = TfidfVectorizer()
            product_vectors = tfidf_product.fit_transform(data.Product).toarray()

            tfidf_description = TfidfVectorizer()
            description_vectors = tfidf_description.fit_transform(data.Description).toarray()

            product_similarity_matrix = cosine_similarity(product_vectors)
            description_similarity_matrix = cosine_similarity(description_vectors)

            # For prices and char count
            normalized_prices = data.Price.values.reshape(1, -1)
            normalized_char_count = data.char_count.values.reshape(1, -1)

            scaler = Normalizer() 
            normalized_prices = scaler.fit_transform(normalized_prices)
            normalized_char_count = scaler.fit_transform(normalized_char_count)

            normalized_prices = cosine_similarity(normalized_prices)
            normalized_char_count = cosine_similarity(normalized_char_count)

            # Combined Similarity with weights
            combined_similarity_matrix = (weight_product * product_similarity_matrix) + (weight_description * description_similarity_matrix) + (weight_prices * normalized_prices) + (weight_char_count * normalized_char_count)

            return combined_similarity_matrix

        combined_similarity_matrix = get_similarity_matrix(
            weight_product = 0.4,
            weight_description = 0.3,
            weight_prices = 0.2,
            weight_char_count = 0.1
        )
        
        def result(prod_id):
            i = combined_similarity_matrix[prod_id]
            a = i.argsort()[::-1][1:4]
            b = sorted(i)[::-1][1:4]

            recs = []
            for j,k in zip(a,b):
                rec = {}
                rec['id'] = j
                rec['sim_score'] = k
                rec['relevant'] = pure_data.category[j] == pure_data.category[prod_id]
                recs.append(rec)

            return recs

        data['result'] = data.id.map(result)
        
        def three_similar_product(prod_id):
            prods = []
            for i in data.result[prod_id]:
                prods.append(i['id'])

            return prods

        self.similar_products =  three_similar_product(prod_id)
        
    def improve(self, key, prod_id):
        pure_data = self.pure_data.copy()
        
        message = 'perbaiki deskripsi produk berikut sehingga menarik bagi pembeli namun mempunyai informasi yang padat \n\n'
        message = message+pure_data.Description[prod_id]
        
        openai.api_key = key

        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[{'role':'user', 'content':f'{message}'}],
          temperature=0,
          max_tokens=1024
        )
        
        return response['choices'][0]['message']['content']