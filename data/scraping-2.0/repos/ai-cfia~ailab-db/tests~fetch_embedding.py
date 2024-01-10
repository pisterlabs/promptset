
import openai
import os 
import sys
import dotenv
dotenv.load_dotenv()

openai.api_type = "azure"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = f"https://{os.environ['AZURE_OPENAI_SERVICE']}.openai.azure.com"
openai.api_version = "2023-05-15"

def fetch_embedding(text):
    """Fetch embedding for a list of tokens from the Microsoft Azure OpenAI API"""
    response = openai.Embedding.create(
        input=text,
        engine="ada"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

if __name__ == '__main__':
    text = " ".join(sys.argv[1:])
    if len(text) == 0:
        print('Please provide a text to embed')
        raise SystemExit
    print(fetch_embedding(text))