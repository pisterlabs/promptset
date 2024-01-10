from ChatGPT.OpenAIEmbeddingWrapper import OpenAIEmbeddingWrapper

if __name__ == '__main__':
    embeddingWrapper = OpenAIEmbeddingWrapper()
    embeddingWrapper.init_embeddings_from_csv('data/data_clean.csv')
