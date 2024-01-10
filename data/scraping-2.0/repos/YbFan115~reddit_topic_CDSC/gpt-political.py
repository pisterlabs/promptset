import openai
import os
import pandas as pd
import time

# Set your OpenAI API key
api_key = "key"
openai.api_key = api_key

# Load your labeled dataset
# Replace 'your_labeled_data.csv' with the path to your labeled dataset
df = pd.read_csv('gun_regression.csv').iloc[:5000]

# Define the initial and final index for slicing
initial_index = 0
final_index = 1000  # Set the initial and final index range

while final_index <= len(df):
    # Load a slice of your labeled dataset
    df_slice = df.iloc[initial_index:final_index]

    # Define a function to classify comments using ChatGPT
    def classify_comments(comments):
        results = []
        for comment in comments:
            messages = [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": f"Is the following Reddit comment discussing public gun law/gun control \
                    gun & politics (including 'dems' or democrats or 'repubs' or republicans) / mass shooting / gun ownership issues? To only answer 'yes' or 'no'. Comment: {comment}"}
                ]

            # Use ChatGPT to generate a response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=1,
                timeout=10
            )

            # Check if ChatGPT's response indicates that the comment discusses gun control issues
            # You can adjust this condition based on ChatGPT's response format
            if "yes" in response.choices[0].message['content'].lower():
                results.append(1)  # Comment discusses public gun control issues
            else:
                results.append(0)  # Comment does not discuss public gun control issues

        return results

    # Split the dataframe slice into chunks of 50 lines
    chunk_size = 50
    num_chunks = len(df_slice) // chunk_size

    results = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_slice.iloc[start:end]
        comments = chunk['comment_text'].tolist()
        results.extend(classify_comments(comments))

    # Add the 'political' column to the dataframe slice
    df_slice['political'] = results

    # Keep only the 'comment_text' and 'political' columns in the dataframe slice
    df_slice = df_slice[['id', 'link_id', 'parent_id', 'subreddit', 'score', 'polisub', 'comment_text', 'author', 'repub_score',
           'repub_normalized', 'length_2', 'length_3', 'length_4', 'political']]

    # Save the updated dataframe slice with the 'political' column
    df_slice.to_csv(f'gun_slice_{initial_index}_{final_index}.csv', index=False)

    # Update the initial and final index for the next iteration
    initial_index = final_index
    final_index += 1000  # Increment the range for the next slice

    # Sleep for 10 minutes before the next iteration
    time.sleep(600)

# The loop will continue until the final index exceeds the length of the dataset

