import openai
import pickle

openai.api_key = "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"


def get_embedding(text, model="text-similarity-davinci-001"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


labels = ["Law - Court - Rights - Constitution", "Philosophical Thoughts - personal thoughts - religion", "Health and fitness",
                    "Governance and Politics", "Arts, Culture and literature", "Business, Finance and Economy",
                    "Entertainment - Films, Movies and TV Shows, Music Album and Concert - Movie Celebrities", "PC Gaming, Xbox, Playstation and Online Gaming", "Lifestyle, Hobbies, Food and Travel", "Science and Technology",
                    "Social Media", "Sports", "World affairs, Climate, Social Issues and Geo-Politics"]

for label in labels:
    output = get_embedding(label)
    with open(f'./embeddings/topic_embeddings/{label}.pkl', 'wb') as file:
        pickle.dump(output, file)
        print(f'{label} - Success')

