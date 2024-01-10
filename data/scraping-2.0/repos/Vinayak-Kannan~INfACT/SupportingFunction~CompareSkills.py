import openai
import pandas as pd
from dotenv import dotenv_values
from openai.embeddings_utils import get_embedding, cosine_similarity


def compare_skills():
    # Import data from csv
    MIT = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/MIT_Fall2023_v1/SkillOutputv2.csv')
    Columbia = pd.read_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/SkillOutputv2.csv')
    config = dotenv_values("/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/.env")

    openai.api_key = config.get("SECRET_KEY")


    MIT['Similar Skill'] = ''
    MIT['Embedding'] = ''
    Columbia['Similar Skill'] = ''
    Columbia['Embedding'] = ''

    embeddings = []
    for text in Columbia['Skill']:
        if text is None or text ==  " " or len(text) == 0:
            embeddings.append(None)
            continue
        embedding_model = "text-embedding-ada-002"
        print(text)
        value = get_embedding(text, engine=embedding_model)
        embeddings.append(value)

    Columbia['Embedding'] = embeddings

    embeddings = []
    for text in MIT['Embedding']:
        if text is None or text ==  " " or len(text) == 0:
            embeddings.append(None)
            continue
        embedding_model = "text-embedding-ada-002"
        print(text)
        value = get_embedding(text, engine=embedding_model)
        embeddings.append(value)

    MIT['Embedding'] = embeddings

    # Find the value in the Skill column in MIT and Columbia that has at least 0.9 cosine similarity using OpenAI embedding API
    # Create a new column in MIT and Columbia called 'Similar Skill' that stores the similar skill
    # Create a new column in MIT and Columbia called 'Similar Skill' that stores the similar skill
    for i, row in enumerate(MIT['Embedding']):
        similarities = []
        for j, row2 in enumerate(Columbia['Embedding']):
            if row is not None and row2 is not None:
                # Calculate the cosine similarity between the two rows
                similarities.append([cosine_similarity(row, row2), Columbia.loc[j, 'Skill'], j])

        # Filter similarities to only include values that are greater than 0.8
        similarities = [x for x in similarities if x[0] >= 0.9]

        if len(similarities) > 0:
            word = similarities[0][1]
            MIT.loc[i, 'Similar Skill'] = word
            continue


    for i, row in enumerate(Columbia['Embedding']):
        similarities = []
        for j, row2 in enumerate(MIT['Embedding']):
            if row is not None and row2 is not None:
                # Calculate the cosine similarity between the two rows
                similarities.append([cosine_similarity(row, row2), MIT.loc[j, 'Skill'], j])

        print(similarities)
        # Filter similarities to only include values that are greater than 0.8
        similarities = [x for x in similarities if x[0] >= 0.9]


        if len(similarities) > 0:
            word = similarities[0][1]
            Columbia.loc[i, 'Similar Skill'] = word
            continue


    # Save columbia and MIT to csv
    MIT.to_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/MIT_Fall2023_v1/comparison.csv', index=False)
    Columbia.to_csv(f'/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/comparison.csv', index=False)