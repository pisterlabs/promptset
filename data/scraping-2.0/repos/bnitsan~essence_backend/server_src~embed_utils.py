class EmbedUtils:
    def __init__(self):
        import openai
        openai.api_type = 'openai'

    def embed(self, input):
        import openai
        print(openai.api_type)
        openai.Embedding.create(input, model='ada-002')

embed_utils = EmbedUtils()
