import pandas as pd

# import openai
# # import key
import pprint
# import json
# import os

df = pd.read_csv('Data/reviews.csv') 

def get_overall_sentiment():

    # set your api key
    openai.api_key = "<secret API Key>"

    # compile reviews from the scraped real-time dataset

    # Load the CSV file into a DataFrame
    df = pd.read_csv('../Data/reviews.csv') 


    # Concatenate the reviewDescription column into a single list of numbered reviews
    reviews = [f"{i+1}. \"{review}\"\n" for i, review in enumerate(df['reviewDescription'])]

    # Join the reviews list into a single string
    concatenated_reviews = ''.join(reviews)

    # Print the result
    print(concatenated_reviews)


    # prompt GPT-3
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Decide whether a tweet sentiment is positive, neutral or negative. Return the percentage of reviews that are positive, neutral and negative. Also tell me the overall sentiment as the most occuring sentiment(positive, negative or neutral). I only want statistics so be to the point." + concatenated_reviews,
        temperature=0,
        max_tokens=40,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )


    pprint.pprint(response)

    
    overall_sentiment = response.choices[0].text.strip()

    # Save the overall sentiment to a file
    with open('overall_sentiment.txt', 'w') as file:
        file.write(overall_sentiment)

    return overall_sentiment

    # Usage
if __name__ == "__main__":
    overall_sentiment = get_overall_sentiment()
    print("Overall Sentiment:", overall_sentiment)