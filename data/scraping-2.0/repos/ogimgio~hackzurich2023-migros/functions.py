import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import re 
from vertexai.language_models import TextEmbeddingModel

from google.cloud import aiplatform

aiplatform.init(project='hackzurich23-8268')



## Load libraries

load_dotenv()  # take environment variables from .env.
#API_KEY = os.environ['API_KEY']
API_KEY = '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'

#model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@latest")

llm= OpenAI(openai_api_key=API_KEY)

meal_template = PromptTemplate(
    input_variables=["recipes"],
    template="Give me the ingredients of: {recipes} and each ingredient in one line",
)

llm_chain = LLMChain(prompt=meal_template, llm=llm)


def get_ingredients_for_recipe_from_llm(question):
    """This function returns the ingredients of a recipe by using the llm model"""

    ingredients=llm_chain.run(question)

    lines = [line for line in ingredients.splitlines() if line.strip()]
    # Join the non-empty lines back into a single string
    ingredients_list=[]
    ingredients_string= '\n'.join(lines)
    for ingredient in ingredients_string.splitlines():
        match = re.search(r'\((.*?)\)', ingredient)
        if match:
            example_text = match.group(1)  # Get the text inside the parentheses
            example_list = [topping.strip() for topping in example_text.split(',')]
            ingredients_list.append(example_list)
            
        else:
             ingredients_list.append(ingredient)
        
    
    return ingredients_list





def aggregate_data():
    """Aggregation of sample data"""
    df = pd.read_csv("Migros_case/Shoppin_Cart/trx_202202.csv")
        # Group by user_id and product_id and sum the 'menge'
    grouped = df.groupby(['KundeID', 'ArtikelID']).Menge.sum()
    
    # Convert the groupby object to a list of triplets
    triplets = [(user, product, amount) for (user, product), amount in grouped.items()]
    
    agg_df = pd.DataFrame(triplets, columns=['user_id', 'product_id', 'amount'])
    agg_df.to_csv("triplets.csv", index=False)
    return



def create_similarity_matrix():
    """
    This function create the similarity matrix by performing matrix factorization, returning a matrix with similarity across users (depending on the products bought),
    which is the definition of collaborative-filtering
    """
    agg_df = pd.read_csv('Migros_case/triplets.csv')
    
    R_df = agg_df.pivot(index = 'user_id', columns ='product_id', values = 'amount').fillna(0)
    R = R_df.values
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    from scipy.sparse.linalg import svds
    U, sigma, Vt = svds(R_demeaned, k = 100)
    sigma = np.diag(sigma)
    original_values = R_df.copy()

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    all_user_predicted_ratings[all_user_predicted_ratings < 0] = 0
    predicted_df = pd.DataFrame(all_user_predicted_ratings, index=R_df.index, columns=R_df.columns)

    mask = (original_values == 0)
    original_values[mask] = predicted_df[mask]
    final_predictions = original_values

    final_predictions.to_pickle('Migros_case/predicted_triplets.pkl')


    similarity_matrix = cosine_similarity(final_predictions)
    similarity_df = pd.DataFrame(similarity_matrix, index=final_predictions.index, columns=final_predictions.index)
    np.fill_diagonal(similarity_df.values, -1)
    similarity_df.to_pickle('Migros_case/similarity_df.pkl')


def create_sustainable_popular_dataset():
    """This function creates the notion of popularity across products"""
    score_df = pd.read_csv('Migros_case/M-Check_packaging.csv', sep=";",encoding='latin-1')
    # drop rows that only contains NaN
    score_df.dropna(how='all', inplace=True)
    #rename column calle dproduct number to product id
    score_df.rename(columns={'Product_number': 'product_id'}, inplace=True)


    # set product id as int
    score_df['product_id'] = score_df['product_id'].astype('string')

    # red triplets.csv  (user_id,product_id,amount) and get most popular products (not based on the amount but the number product_id occurs) and normalise the count from 0 to 1
    triplets_df = pd.read_csv('Migros_case/triplets.csv')
    grouped_triplets_df = triplets_df.groupby('product_id').count().reset_index()
    grouped_triplets_df['amount'] = grouped_triplets_df['user_id'].apply(lambda x: x / grouped_triplets_df['user_id'].sum())
    grouped_triplets_df.drop('user_id', axis=1, inplace=True)

    # set proiduct id as string
    grouped_triplets_df['product_id'] = grouped_triplets_df['product_id'].astype('string')

    # emrge it with score_df
    score_df = score_df.merge(grouped_triplets_df, on='product_id', how='left').fillna(0)
    #rename amount to popularity
    score_df.rename(columns={'amount': 'popularity'}, inplace=True)
    score_df.to_csv("Migros_case/Score_popularity.csv")

def get_product_from_similar_user(sorted_user_similarities, triplets_df,userID,num_users=1):
    """This function returns the products of the most similar num_users (based on collaborative filtering)"""
    # get num_users most similar users
    most_similar_users = sorted_user_similarities.iloc[0:num_users].index.tolist()

    print('Similar user: ',most_similar_users)

    # get all product of userId and all products of user_row_number (they are in triplets_df)
    most_similar_user_products = triplets_df[triplets_df['user_id'].isin(most_similar_users)]['product_id'].tolist()

    return most_similar_user_products


def recommend_similar_users_sustainable_products(predictions_df, userID, score_df, triplets_df, num_recommendations=5):
    """This function returns the products of the most similar user and then sort them by sustainability score"""
    
    # Get and sort the user's predictions
    sorted_user_similarities = predictions_df.loc[userID].sort_values(ascending=False)

    # get products of most similar user by looking at 
    most_similar_user_products = get_product_from_similar_user(sorted_user_similarities, triplets_df,userID,num_users=1)

    # get all product of userId and all products of user_row_number (they are in triplets_df)
    userID_products= triplets_df[triplets_df['user_id'] == userID]['product_id'].tolist()
    # get products that are in most_similar_user_products but not in userID_products
    products_to_recommend = np.setdiff1d(most_similar_user_products, userID_products)

    
    #sort by sustainabiltiy score (the score is in score_df)
    products_to_recommend = products_to_recommend[np.argsort(score_df[score_df['product_id'].isin(products_to_recommend)]['sust_score'])]

    if products_to_recommend.size == 0:
        # if there is no product, look into the top 3 similar users
        most_similar_user_products = get_product_from_similar_user(sorted_user_similarities, triplets_df,userID,num_users=3)
        products_to_recommend = np.setdiff1d(most_similar_user_products, userID_products)
        products_to_recommend = products_to_recommend[np.argsort(score_df[score_df['product_id'].isin(products_to_recommend)]['sust_score'])]

        if products_to_recommend.size == 0:
        
            # just give popular sustainable products (popular based on popularity, sustainable based on 'mcheckpackaging). Suistanable must have priority
            products_to_recommend = score_df.sort_values(['sust_score', 'popularity'], ascending=[False, False])['product_id'].tolist()
        
    # return top 5
    return products_to_recommend[:num_recommendations]


def calculate_allproduct_embedding():
    """This function calculates the embedding of all products and save it in a dataframe"""
    products = pd.read_csv("Migros_case/products.csv",index_col=0)
    products.index = products.index.astype(str)
    products_names = products.name.tolist()
    

    embeddings = [np.mean([np.array(embedding.values) for embedding in model.get_embeddings([product_name])], axis=0) for product_name in tqdm(products_names,total=len(products_names))]

    # return dataframe with column product_idx, product_name, product_embedding
    pd.DataFrame({'product_idx':products.index,'product_name':products_names,'product_embedding':embeddings}).to_pickle("Migros_case/product_embeddings.pkl")

def get_similar_product_from_given(choosenID):
    """This function returns the most similar product based on the embedding of the choosen product, + makes sure to return a product which is more sustainable"""
    scores_df = pd.read_csv('Migros_case/sust_score.csv')
    # as str
    scores_df.product_id = scores_df.product_id.astype(str)

    embeddings = pd.read_pickle("Migros_case/product_embeddings.pkl")
    # Randomly pick a product
    # Compute embedding of chosen product
    choosen = embeddings[embeddings['product_idx'] == choosenID]['product_embedding'].values[0]
    if choosen is None:
        print("User id not found")
        return None
    
    # remove the embedding with id choosenId from embeddings (embedding is a dataframe with value product_idx and product_embedding)

    # select a random embedding
    other_embeddings = embeddings[embeddings['product_idx'] != choosenID]['product_embedding'].values
    similarities = [util.pytorch_cos_sim(choosen, embedding) for embedding in other_embeddings]
    similarities = np.array([tensor.item() for tensor in similarities])
    # get top 10 most similar products idx
    top10_idx = np.argsort(similarities)[-10:]
    # get top1_idx 
    top10_idx = embeddings[embeddings['product_idx'] != choosenID]["product_idx"].values[top10_idx]

    currentRating = scores_df[scores_df['product_id'] == choosenID][['product_id','sust_score']]
    suggestedRating = scores_df[scores_df['product_id'].isin(top10_idx)][['product_id','sust_score']]
    if suggestedRating.empty:
        print("No similar product found")
        return None

    currentRating_value = currentRating["sust_score"].values[0]
    
    suggestedRating_values = suggestedRating["sust_score"].values.tolist()

    # Now we have a current rating and 10 suggested ratings, we need to find the first one that is better than the current one and return product id + id
    # if there is no better rating, return None

    print(currentRating_value)
    for idx,rating in enumerate(suggestedRating_values):
        if rating > currentRating_value:
            return suggestedRating["product_id"].values[idx]
        
    return None


def get_similar_sustainable_product_from_text(text_name):
    """This function returns the most similar product based on the embedding of a text"""
    scores_df = pd.read_csv('Migros_case/sust_score.csv')
    # as str
    scores_df.product_id = scores_df.product_id.astype(str)

    embeddings = pd.read_pickle("Migros_case/product_embeddings.pkl")
    # Randomly pick a product
    # Compute embedding of chosen product
    choosen = np.mean([np.array(embedding.values) for embedding in model.get_embeddings([text_name])], axis=0)

    similarities = [util.pytorch_cos_sim(choosen, embedding) for embedding in embeddings['product_embedding'].values]
    similarities = np.array([tensor.item() for tensor in similarities])
    # get top 10 most similar products idx
    top10_idx = np.argsort(similarities)[-20:]
    # get top1_idx 
    top10_idx = embeddings["product_idx"].values[top10_idx]

    # return products in order of sust_score
    suggestedRating = scores_df[scores_df['product_id'].isin(top10_idx)][['product_id','sust_score']]
    
    # order by sust_score (higher is best)
    suggestedRating = suggestedRating.sort_values(by=['sust_score'], ascending=False)[:10]

    return suggestedRating.to_dict()



    