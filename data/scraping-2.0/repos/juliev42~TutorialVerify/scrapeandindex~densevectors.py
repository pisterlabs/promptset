import openai

openai.api_key = os.environ['OPENAI_API_KEY']

def get_dense_vector(text):
    response = openai.Embedding.create(
    input="Your text string goes here",
    model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']