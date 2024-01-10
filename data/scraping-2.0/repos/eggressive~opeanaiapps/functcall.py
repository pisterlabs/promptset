"""_Review classification using function calling_
"""
import os
import openai
import requests
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]

DATA_URL = ("https://github.com/blueprints-for-text-analytics-"
            "python/blueprints-text/raw/master/data/amazon-"
            "product-reviews/reviews_5_balanced.json.gz")
LOCAL_DATA = "/home/dimitar/openai/apps/data/reviews_5_balanced.json.gz"
EXPECTED_SIZE = 16649530

if os.path.exists(LOCAL_DATA):
    local_size = os.path.getsize(LOCAL_DATA)
    if local_size == EXPECTED_SIZE:
        print(f"{LOCAL_DATA} already exists with expected size {EXPECTED_SIZE}, skipping download")
    else:
        print(f"Existing file {LOCAL_DATA} size {local_size}"
              f"does not match expected size {EXPECTED_SIZE}, re-downloading...")

        with requests.get(DATA_URL, stream=True, timeout=30) as r:
            r.raise_for_status()

            with open(LOCAL_DATA, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

else:
    print(f"{LOCAL_DATA} does not exist, downloading...")

    with requests.get(DATA_URL, stream=True, timeout=30) as r:
        r.raise_for_status()

        with open(LOCAL_DATA, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# create data frame from the json file
df = pd.read_json(LOCAL_DATA, lines=True)
df = df.drop(columns=['reviewTime','unixReviewTime'])
df = df.rename(columns={'reviewText': 'text'})
df.sample(5, random_state=12)

# Assigning a new [1,0] target class label based on the product rating
df['sentiment'] = 'OTHER'
df.loc[df['overall'] > 3, 'sentiment'] = 'POSITIVE'
df.loc[df['overall'] < 3, 'sentiment'] = 'NEGATIVE'

# Removing unnecessary columns to keep a simple dataframe
df.drop(columns=['overall', 'reviewerID', 'summary'],
        inplace=True)
df.sample(5)

REVIEW_NUMBER = 65
review_text = df['text'].iloc[REVIEW_NUMBER]
print (review_text)

completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": review_text}],
    functions=[
    {
        "name": "forward_review_using_sentiment",
        "description": "This function forwards the review to an appropriate customer service agent based on whether it is POSITIVE or NEGATIVE.",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "description": "The sentiment of the review which is POSITIVE if the customer liked the product and is NEGATIVE if the customer did not like the product",
                },
                "unit": {"type": "string"},
            },
            "required": ["sentiment"],
        },
    }
    ],
    function_call={"name": "forward_review_using_sentiment"}
    )

print(completion)
