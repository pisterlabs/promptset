"""
This scripts creates the embeddings for all the data.

It is mainly split into two parts:
1. Prepare the raw data for embedding (chunking, cleaning, etc.)
2. Create the embeddings for each category and subcategory

Importantly, in the second part, we are storing everything in a specific way
so that we can easily access the embeddings for each category and subcategory,
as well as the underlying data that was used to create the embeddings.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import pickle

# Create all the embeddings for the categories and subcategories
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Load the data - two spots depending on where 3_prep_for_embeddings.py was run. (prefer to run this from /text_search)
try: 
    categorized_data = pd.read_csv("../data_collection/data/chunked_data.csv")
except:
    categorized_data = pd.read_csv("chunked_data.csv")

categorized_data = categorized_data.loc[:, ['category', 'subcategory', 'chunked_data']]
df = categorized_data.rename(columns={'chunked_data': 'data'})

# Line of code from `train_subcat_classifier.py`
categories_with_subcategories = df['category'].unique()[df.groupby(["category"])['subcategory'].nunique() > 4]

# Create embeddings for all the categories and subcategories and store them in a dictionary
embeddings = {}
data = {}

for category in df['category'].unique():
    print("Writing embeddings for category: ", category, "...")
    category_df = df[df['category'] == category]

    if category in categories_with_subcategories:
        # first get the embeddings for the main category
        data_to_embed = category_df[category_df['subcategory'].isnull()]['data'].to_list()
        if data_to_embed != []:
            embeddings[category] = model.encode(data_to_embed, normalize_embeddings=True)
            data[category] = data_to_embed

        # split the data into subcategories
        subcategories = category_df[category_df['subcategory'].notnull()]['subcategory'].unique()
        for subcategory in subcategories:
            data_to_embed = category_df[category_df['subcategory'] == subcategory]['data'].to_list()
            if data_to_embed != []:
                embeddings[f"{category}-{subcategory}"] = model.encode(data_to_embed, normalize_embeddings=True)
                data[f"{category}-{subcategory}"] = data_to_embed
    else:
        data_to_embed = category_df['data'].to_list()
        if data_to_embed != []:
            embeddings[category] = model.encode(data_to_embed, normalize_embeddings=True)
            data[category] = data_to_embed

data = {'embeddings': embeddings, 'data': data}

# Save the embeddings
with open("embeddings.pickle", "wb") as f:
    pickle.dump(data, f)

print("\nComplete!")