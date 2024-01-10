import openai


class OpenAIClient:
    def __init__(self, api_key):
        # Init OpenAI
        openai.api_key = api_key

        # Set openai on class. Consumers of this class can use it to call openai API directly.
        self.openai = openai

    def create_embeddings(self, text, model="text-embedding-ada-002"):
        response = self.openai.Embedding.create(
            input=text,
            model=model
        )
        embeddings = response['data'][0]['embedding']
        return embeddings

    def create_completion(self, payload):
        completion = self.openai.Completion.create(**payload)
        return completion
