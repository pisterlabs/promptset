import asyncio
import openai
import tiktoken
import openai_async
import pandas as pd
import random

with open('../secretkey.txt', 'r') as f:
    secret = f.readline()

df = pd.read_csv('../datasets/ratings/sampled_reviews_no_skew.csv')
df = df.sample(n=1000)
samples = len(df)


enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
possible_nums=range(1,6) # int from 1 to 11
possible_tokens = {f"{enc.encode(f'{i}')[0]}": 100 for i in possible_nums} #coerces model to generate a number from 1 to 10



async def generateRating(row):
    review = row['Review']
    true_rating = row['Rating']
    row_num = row.name
    prompt = f"The following is a review from a customer who stayed at a hotel. Please rate the hotel on a scale from 1 to 5, with 1 being the worst and 10 being the best. Number only. \n\nReview: {review} \n\nRating: "

    for attempt in range(5):  # Will try up to 5 times
        try:
            completion = await openai_async.chat_complete(
                secret,
                timeout=60,
                payload={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1,
                    "logit_bias": possible_tokens
                },
            )
            openai_rating = completion.json()['choices'][0]['message']['content'].strip().lower()
            return (openai_rating, true_rating, row_num)
        except Exception as e:
            if attempt < 4:  # No need to sleep on the last attempt
                backoff_time = 5 * (attempt + 1) + random.uniform(-5, 5)
                await asyncio.sleep(backoff_time)
            else:
                return (None, true_rating, row_num)

async def run_concurrent_calls():
    semaphore = asyncio.Semaphore(100)

    async def bounded_generateRating(row):
        async with semaphore:
            return await generateRating(row)

    tasks = [bounded_generateRating(row[1]) for row in df.iterrows()]

    results = await asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    ratings = loop.run_until_complete(run_concurrent_calls())

    results_df = pd.DataFrame(ratings, columns=['openai_rating', 'true_rating', 'original_dataset_row_number'])
    results_df.to_csv('ratings_comparison_no_skew.csv', index=False)
    print("Saved ratings to 'ratings_comparison.csv'")
