import os
import sys
import asyncio
import aiohttp
from asyncio import Queue

import pandas as pd
from tqdm import tqdm
import openai

RATE_LIMIT = 50  # Number of tasks per minute

async def get_anonymized_review(sem: any, session: any, text: str, clinic_name: str, api_key: str, model: str = 'gpt-3.5-turbo') -> str:
    async with sem:
        openai.api_key = api_key
        clinic_name = clinic_name.replace("_", " ")  # replace underscores with spaces
        message = f"Anonymize this review by replacing city/states, business/clinic names, and individual name(s) (including dr. or pronouns) with X. For reference, the business name to anonymize is {clinic_name}.\n\n{text}."
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are anonymizing review text for author privacy."},
                    {"role": "user", "content": message}
                ]
            }
        )
        response = await response.json()
        if 'choices' not in response:
            print(f"Error: {response}")
            return None  # Return None or handle error appropriately
        return response['choices'][0]['message']['content']

async def rate_limiter(task_queue: Queue, sem, session, rows, api_key):
    for idx, row in tqdm(rows.iterrows(), total=rows.shape[0]):
        await task_queue.put(asyncio.ensure_future(get_anonymized_review(sem, session, row['Review_Text'], row['Clinic_Name'], api_key)))
        if (idx + 1) % RATE_LIMIT == 0:  # Wait for 60 seconds every n tasks
            await asyncio.sleep(60)

async def main(input_csv: str, output_csv: str, api_key: str) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    # Set up a semaphore for rate limiting
    sem = asyncio.Semaphore(RATE_LIMIT)

    # Get rows that still need to be processed
    df_out = pd.read_csv(output_csv)
    unprocessed_rows = df_out[df_out['Anonymized_Review_Text'].isna() | (df_out['Anonymized_Review_Text'].str.strip() == "")]
    print(f'Number of unprocessed rows: {unprocessed_rows.shape[0]}')

    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        task_queue = asyncio.Queue()
        asyncio.create_task(rate_limiter(task_queue, sem, session, unprocessed_rows, api_key))

        for idx in tqdm(range(unprocessed_rows.shape[0])):
            task = await task_queue.get()
            resp = await task

            df_out.loc[unprocessed_rows.index[idx], 'Anonymized_Review_Text'] = resp

            # Save progress every 20 reviews
            if idx % 20 == 0:
                df_out.to_csv(output_csv, index=False)
                print(f'saving at idx: {idx}')

        df_out.to_csv(output_csv, index=False)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python anonymize_reviews.py input.csv output.csv api_key")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    api_key = sys.argv[3]

    print(f'input csv: {input_csv}')
    print(f'output csv: {output_csv}')

    asyncio.run(main(input_csv, output_csv, api_key))
