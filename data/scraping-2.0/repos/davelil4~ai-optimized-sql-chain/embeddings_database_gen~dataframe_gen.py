import pandas as pd
import os
# from transformers import pipeline
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = "https://api.openai.com/v1"

model = "juierror/text-to-sql-with-table-schema"

def init_schemata_df():
    
    data = {
        'schemata': [],
        'context': [],
    }
    
    for filename in os.listdir(os.path.join(os.getcwd(), 'schemata')):
        if (".txt" in filename):
            f = open(os.path.join(os.getcwd(), f'schemata/{filename}'), 'r')
            data['schemata'].append(filename)
            data['context'].append(f.read())
            f.close()
    
    df = pd.DataFrame(data=data)
    
    df.to_csv(os.path.join(os.getcwd(), 'dataframe_csvs/schemata_df.csv'), index=False)
    
def apply_schemata_embeddings_openai():
    df = pd.read_csv(os.path.join(
        os.getcwd(),
        'dataframe_csvs/schemata_df.csv'
    ), dtype=str)
    
    df['context'] = df.context.astype(str)
    
    try:
        df['embedding'] = df.context.apply( lambda x: 
            # get_embedding(x, model='text-embedding-ada-002')
            openai.Embedding.create(model='text-embedding-ada-002',
                                    input=x).data[0].embedding
            )
    except Exception as e:
        print(e)
    
    print(df)
    
    df.to_csv(os.path.join(os.getcwd(), 'dataframe_csvs/schemata_embeddings_openai_df.csv'), index=False)

def apply_schemata_embeddings_hf():
    df = pd.read_csv(os.path.join(
        os.getcwd(),
        'dataframe_csvs/schemata_df.csv'
    ))
    
    client = InferenceClient(model=model, token=os.environ['TOKEN'])
    
    extract = client.text_generation
    
    try:
        df['embedding'] = df['context'].apply( lambda x: extract(x))
    except Exception as e:
        print(e)
    # print(e)
    
    df.to_csv(os.path.join(os.getcwd(), 'dataframe_csvs/schemata_embeddings_hf_df.csv'), index=False)
    
if __name__ == "__main__":
    # print(pd.read_csv(os.path.join(os.getcwd(), 'dataframe_csvs/schemata_embeddings_hf_df.csv')))
    apply_schemata_embeddings_openai()