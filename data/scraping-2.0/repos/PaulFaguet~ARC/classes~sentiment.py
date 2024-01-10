import openai 
import pandas as pd 
from dotenv import load_dotenv
import time
import streamlit as st
from nltk import word_tokenize
from math import ceil
import asyncio


load_dotenv()

class Sentiment:
    def __init__(self, df: pd.DataFrame, export: bool = False, execution_time: bool = False):
        self.df = df
        self.export = export
        self.execution_time = execution_time
        
        self.time_duration = None
    
    
    def _question_davinci_model(self, review: str):
        prompt = f"D'après l'avis suivant, détermine le sentiment exprimé parmi les qualificatifs suivants : positif, neutre ou négatif.\n\nAvis : {review}\n\nSentiment :"
        
        # response = openai.Completion.create(
        #     # model='text-davinci-003',
        #     model='text-curie-001',
        #     prompt=prompt,
        #     temperature=0.1,
        #     # max_tokens=self._tokenizer(review),
        #     max_tokens=1000
        # )

        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "Tu es un expert SEO."},
                {"role": "user", "content": prompt},
            ]
        )
        
        response = response.choices[0].message.content.replace('.', '').lower()
        
        return response

    def _process_row(self, row: pd.Series):
        return self._question_davinci_model(row['Review'])
    
    def _formate_df_headers(self):
        self.df = self.df[['Last Updated Date', 'ID', 'Review']]
        
        return
    
    def _subset_df(self):
        df = self.df
        
        if len(df) > 1500:
            dataframes = []
            
            subsets = round(len(df) / 1500)
            
            for i in range(subsets):
                dataframes.append(df[i*1500:(i+1)*1500])
            
            return dataframes

        else:
            return [df]
        
    def _tokenizer(self, text: str):
        nbr_of_words = len(word_tokenize(text))
        
        if nbr_of_words:
            result = nbr_of_words / 0.75
            result += result / 2 
            
            return round(result)

        else:
            return 1000
        
    
    def run(self):
        progress_bar = st.progress(0)
        
        # ne garde que les colonnes suivantes : 'Last Updated Date', 'ID', 'Review'
        self._formate_df_headers()
        
        if self.execution_time:
            start = time.time()
        
        # results = [self._process_row(row) for _, row in self.df.iterrows()]
        results = []
        for index, row in self.df.iterrows():
            results.append(self._process_row(row))
            progress_bar.progress(ceil((index+1)/len(self.df) * 100))
        
        # dataframes = self._subset_df()
    
        # if len(dataframes) > 1:
        #     # use asyncio to multiprocess the dataframes
        #     loop = asyncio.get_event_loop()
        #     results = loop.run_until_complete(asyncio.gather(*[self._process_row(row) for _, row in self.df.iterrows()]))
        #     merged_df = pd.concat(results)
        #     self.df['Sentiment'] = merged_df['Sentiment']
        #     progress_bar.progress(100)
        # else:
            # results = []
            # for index, row in self.df.iterrows():
            #     results.append(self._process_row(row))
            #     progress_bar.progress(ceil((index+1)/len(self.df) * 100))
            # self.df['Sentiment'] = results
        
        if self.execution_time:
            st.info(f'Temps d\'exécution : {ceil((time.time() - start)/60)} minutes')
            
        self.df['Sentiment'] = results
        
        return