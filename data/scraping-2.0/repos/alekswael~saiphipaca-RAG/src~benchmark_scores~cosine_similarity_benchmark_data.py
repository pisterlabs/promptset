import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings import HuggingFaceInstructEmbeddings
import PyPDF2
import random
import os
import numpy as np

# Don't print out warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(420)

# Set the data path
data_path = "results/"

# Make a folder for the results titled "cosine_similarity"
if not os.path.exists(data_path + "cosine_similarity"):
    os.makedirs(data_path + "cosine_similarity")

# Make a list of the csv files to be used - all .csv files in the data_path folder
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

# Initialize the vectorizer
#vectorizer = TfidfVectorizer()
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Initialize a loop per each csv file
for csv_file in csv_files:

    # Load the data
    df = pd.read_csv(data_path + csv_file)

    # Convert all 'text' values to string
    df['gpt_response'] = df['gpt_response'].astype(str)
    df['model_response'] = df['model_response'].astype(str)

    cosine_similarity_list = []

    # Add a new column to the dataframe, titled 'cosine_similarity'
    df['cosine_similarity'] = ""

    # Loop through dataframe and compare 'answer' and 'model_response' columns using cosine similarity, adding the results to the 'cosine_similarity' column
    for index, row in df.iterrows():

        # Create a corpus containing the two texts
        text = [row['gpt_response'], row['model_response']]

        # Now vectorize the corpus
        #X = vectorizer.fit_transform(text)

        # Using the embeddings
        X = embeddings.embed_query(text[0])
        X = np.array(X)

        Y = embeddings.embed_query(text[1])
        Y = np.array(Y)

        # Compute the cosine similarity between the two texts
        #cos_sim = cosine_similarity(X[0], X[1])[0][0]
        cos_sim = cosine_similarity([X],[Y])[0][0]

        # Append the cosine similarity to the dataframe
        df.at[index, 'cosine_similarity'] = cos_sim

    print(f"Cosine similarity mean for answer pairs in the {csv_file} dataset is: " + str(df['cosine_similarity'].mean()))

    # Save the dataframe as single_paper_results_phi-1_5_cosine.csv in the cosine_similarity folder
    df.to_csv(data_path + "cosine_similarity/" + csv_file, index = False)