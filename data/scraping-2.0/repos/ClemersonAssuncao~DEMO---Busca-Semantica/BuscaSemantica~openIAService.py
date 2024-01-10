
import openai
from openai.embeddings_utils import cosine_similarity
from django.conf import settings
import pandas as pd
from .pdfReader import PdfReader
import numpy as np

class OpenIAService:

    def __init__(self):
        if (settings.EXECUTE_API_OPEN_IA):
            self.FILE_DATA_FRAME = settings.DF_FILE_NAME
        else:
            self.FILE_DATA_FRAME = settings.DEBUG_DF_FILE_NAME
        openai.api_key = settings.OPEN_IA_TOKEN 
        self.EMBEDDING_ENGINE = 'text-embedding-ada-002'

    def __get_data_frame_file(self):
        try:
            df = pd.read_csv(self.FILE_DATA_FRAME)
            return df
        except pd.errors.EmptyDataError:
            return None
        except FileNotFoundError:
            return pd.DataFrame(columns=['id', 'type', 'text'])

    def get_embedding(self,text_to_embed):
        response = openai.Embedding.create(
            model= self.EMBEDDING_ENGINE,
            input=[text_to_embed]
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    
    def search(self, text):
        df = self.__get_data_frame_file()
        if (settings.EXECUTE_API_OPEN_IA): 
            if (settings.EXECUTE_API_OPEN_IA):
                df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
                embedding = self.get_embedding(text)
                df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
        else:
                df['similarities'] = df.id.apply(lambda x: 0.8)
        df = df.sort_values('similarities', ascending=False)
        df = df.drop_duplicates(subset=['id'], keep='first')
        print(df)
            
        return df

    def appendFile(self, instance):
        json_data = PdfReader(instance).get_json_file()
        df_new = pd.DataFrame(json_data)
        if (settings.EXECUTE_API_OPEN_IA):
            df_new['ada_embedding'] = df_new['text'].apply(lambda x: self.get_embedding(x))
        
        df = self.__get_data_frame_file()
        df = df.drop(df[(df['id'] == instance.id) & (df.type == 'file')].index)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(self.FILE_DATA_FRAME, index=False)

    def appendInstance(self, instance):
        json_data = PdfReader(instance).get_json_instance()
        df_new = pd.DataFrame(json_data)
        if (settings.EXECUTE_API_OPEN_IA):
            df_new['ada_embedding'] = df_new['text'].apply(lambda x: self.get_embedding(x))
        df = self.__get_data_frame_file() 
        df = df.drop(df[(df['id'] == instance.id) & ((df.type == 'description') | (df.type == 'name'))].index)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(self.FILE_DATA_FRAME, index=False)
    
        
    def deleteInstance(self, instance):
        df = pd.read_csv(self.FILE_DATA_FRAME)
        df = df.drop(df[df.id == instance.id].index)
        df.to_csv(self.FILE_DATA_FRAME, index=False)

