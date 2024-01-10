import pandas as pd
from openai import OpenAI
import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm  # tqdm is a library to display progress bars in console
import logging
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Code is from https://replit.com/@olafblitz/tuna-asyncio?v=1&ref=blog.langchain.dev#main.py ?
class TokenBucket:
    """
    This class implements the token bucket algorithm which helps in rate limiting.
    The idea is that tokens are added to the bucket at a fixed rate and consumed when requests are made.
    If the bucket is empty, it means the rate limit has been exceeded.
    """

    def __init__(self, rate: int, capacity: int):
        """
        Initialize a new instance of TokenBucket.

        Parameters:
        - rate: The rate at which tokens are added to the bucket.
        - capacity: Maximum number of tokens the bucket can hold.
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = asyncio.get_event_loop().time()

    async def consume(self):
        """
        Consume a token from the bucket. If no tokens are available, it waits.
        """
        while not self._consume():
            await asyncio.sleep(0)
        await asyncio.sleep(1 / self._rate)

    def _consume(self):
        """
        Internal method to consume a token. Refills the bucket based on elapsed time.
        """
        current_time = asyncio.get_event_loop().time()
        elapsed_time = current_time - self._last_refill

        refill_tokens = self._rate * elapsed_time
        self._tokens = min(self._capacity, self._tokens + refill_tokens)
        self._last_refill = current_time

        if self._tokens >= 1:
            self._tokens -= 1
            return True
        return False

# Semaphore is used to limit the number of simultaneous requests.
SEMAPHORE = asyncio.Semaphore(80)
RATE_LIMITER = TokenBucket(80, 80)  # Set rate and capacity both to 100.

logging.basicConfig(filename='openai_api_log.txt', level=logging.INFO)
client = OpenAI()


# Load your CSV file
df = pd.read_csv('job_info.csv')

with open ("example_description.txt", "r") as f:
    example_description=f.readlines()


with open("output_example.txt", "r") as f:
    output_example=f.readlines()

# Define a function to make API calls
async def get_openai_response(prompt):
    # Wait until a token is available
    await RATE_LIMITER.consume()
    async with SEMAPHORE:
        try:
            print("Running API call")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",  # Replace with the model you're using
                messages=[
                    {"role": "system", "content": "You will help creating a training dataset for extracting job description requirements."},
                    {"role": "system", "content": "If there is no data for the specific tag, simply just return \"N/A\"."},
                    {"role": "system", "content": "If experience is either a degree or from a related field, remember to university degree as an alternative in the experience."},
                    {"role": "system", "content": "IMPORT!: Make requirements as concise as possible."},
                    {"role": "system", "content": "Here is an example:"},
                    {"role": "system", "content": f"input: {example_description}"},
                    {"role": "system", "content": f"output: {output_example}"},
                    {"role": "system", "content": "IMPORT!: Some inputs might contain alot of website text, ignore this and focus on the job description."},
                    {"role": "user", "content": f"{prompt}"},
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(e)
            return None



async def main(df):
    df['openai_response'] = ''
    # Iterate through the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        input_text = row['Description']  # Replace with your input column name
        response = await get_openai_response(input_text)
        if response is not None:
            df.at[index, 'openai_response'] = response
        if index % 10 == 0:
            intermediate_file = f'updated_file_intermediate_{index}.csv'
            df.to_csv(intermediate_file, index=False)
            logging.info(f"Intermediate save at row {index} to {intermediate_file}")

    # Save the updated DataFrame to a new CSV file
    df.to_csv('updated_file.csv', index=False)
    logging.info("Final save to updated_file_final.csv")


if __name__ == '__main__':
    asyncio.run(main(df))
