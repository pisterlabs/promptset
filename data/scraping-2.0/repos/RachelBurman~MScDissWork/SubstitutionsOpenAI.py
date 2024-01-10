import openai
import pandas as pd

keys = {}
with open('keys.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split(' = ')
        keys[key] = value

# Now you can access the keys and values like this:
api_key = keys['API KEY']
auth_details = eval(keys['AUTHDETAILS'])
bolt = keys['BOLT']

openai.api_key = api_key

# Load data into a DataFrame
df = pd.read_csv('closest_nodesTEST.csv')

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    ingredient = row['Ingredient']
    closest_nodes = [row[f'Closest Node {i}'] for i in range(1, 6)]  # get all closest nodes
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What are some suitable substitutes for {ingredient} for a vegan from these options: {closest_nodes}?"}
            ]
        )
        # Extracting the response text
        answer = response['choices'][0]['message']['content'] if response['choices'] else "No response"
        print(f"Substitutions for {ingredient} for vegans from {closest_nodes}: {answer}")
    except Exception as e:
        print(f"Error with ingredient {ingredient}: {e}")