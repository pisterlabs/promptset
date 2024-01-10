import openai

openai.api_key = "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"
def get_similar_texts(text:str, k: int = 5):
    response = openai.Embedding.create(input=text,model="text-embedding-ada-002")
    vector = response['data'][0]['embedding']
    print(vector)

get_similar_texts("I like to eat apples and bananas")