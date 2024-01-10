import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict

# Load the data from an Excel file
file_path = 'tr11.xlsx'  # Update the file path to your Excel file
data = pd.read_excel(file_path)

# Format the 'Date' column
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S')

# Ensure 'Customer ID' and 'Product_ID' are treated as categorical variables
data['Customer ID'] = data['Customer ID'].astype('category')
data['Product_ID'] = data['Product_ID'].astype('category')

# Create a user-item matrix
user_item_matrix = data.pivot_table(index='Customer ID', columns='Product_ID', values='Quantity', fill_value=0)

# Load the data into Surprise format
reader = Reader(rating_scale=(0, user_item_matrix.values.max()))
data_surprise = Dataset.load_from_df(data[['Customer ID', 'Product_ID', 'Quantity']], reader)
trainset, testset = train_test_split(data_surprise, test_size=0.25)

# Train an SVD model
model = SVD()
model.fit(trainset)

# Function to get top N recommendations for each user
def get_top_n_recommendations(predictions, n=1):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Test the model and get the top recommendation for each user
predictions = model.test(testset)
top_n_recommendations = get_top_n_recommendations(predictions, n=1)

# Write recommendations to a text file
with open('customer_recommendations.txt', 'w') as file:
    for uid, user_ratings in top_n_recommendations.items():
        product_id, _ = user_ratings[0]
        file.write(f"Customer Id: {uid}, we recommend you to buy Product ID: {product_id}\n")

# Function to fetch product description
def get_product_description(product_id):
    # Assuming 'Description' is the column name for product descriptions
    if product_id in data['Product_ID'].values:
        return data[data['Product_ID'] == product_id]['Description'].iloc[0]
    else:
        return "Description not found."

# Try to import OpenAI's GPT-3 for generating recommendations
import openai
openai.api_key = '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'

# Function to generate a personalized recommendation message using OpenAI's API
def generate_personalized_message(customer_id, product_id):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Write a personalized product recommendation message for a customer. The customer's ID is {customer_id}, and the recommended product ID is {product_id}."}
        ]
    )
    # Extract the message from the response
    message = response.choices[0].message['content']
    return message


# Convert customer_id_input to the appropriate data type before comparison
customer_id_input = input("Please enter a Customer ID to get a personalized recommendation: ")
try:
    # Assuming customer IDs are integers, convert the input to an integer
    customer_id_input = int(customer_id_input)
except ValueError:
    print("Please enter a valid Customer ID.")
    exit()

# Check if the converted customer_id_input is in the top_n_recommendations
if customer_id_input in top_n_recommendations:
    product_id, _ = top_n_recommendations[customer_id_input][0]
    product_description = get_product_description(product_id)
    
    personalized_message = generate_personalized_message(customer_id_input, product_description)
    print(personalized_message)
    with open('enhanced_customer.txt','w') as output_file:
        output_file.write(personalized_message)
else:
    print(f"Customer ID {customer_id_input} not found in the recommendations. Please ensure the Customer ID is correct and try again.")
