from openai import OpenAI
import pandas as pd
import time
import os

hashtag_df = pd.read_csv("item_df.csv", usecols=['id'], sep=',')
prompt = """
Give me a comma-separated list of three short descriptions, that best captures the style of the given clothing.
Use abstract and high-level everyday language keywords rather than nitty-gritty fashion jargons.
I am essentially trying to create hashtag fashion keywords for these clothings, and need your help to create those labels.
So try to generate meaningful keywords, not too simple yet not too detailed.
"""

# def load_api_key(api_file="openai-api.json"):
#     with open(api_file) as f:
#         key = json.load(f)
#     return key["OPENAI_API_KEY"]

client = OpenAI()
client.api_key = os.environ["OPENAI_API_KEY"] # load_api_key()

def process_id_with_retry(id):
    image_url = f"https://tryot.s3.ap-northeast-2.amazonaws.com/item_img/{id}.jpg"
    max_retry_attempts = 3
    retry_delay_seconds = 100  # Adjust this based on your needs

    for attempt in range(max_retry_attempts):
        try:
            print(f"ID: {id}, Attempt: {attempt+1}")
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_message = str(e)
            if "rate limit" in error_message.lower():
                print(f"Rate limit error. Retrying after {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
            else:
                print(f"Other error: {error_message}")
                break  # Break the loop for non-rate-limit errors

    return None  # Return None if all retry attempts fail

# generate hashtags starting from the given id
def generate(idxs):
    # Create a new CSV file for writing
    with open("hashtag.csv", "a") as f:
        # f.write("id, hashtags\n")  # Write header
        count = 0
        for id in hashtag_df["id"][idxs]:
            count += 1
            content = process_id_with_retry(id)
            if content is not None:
                f.write(f"{id}, {content}\n")
                if count % 20 == 0: 
                    print("Writing to file.")
                    f.flush()
                    os.fsync(f.fileno()) # sync all internal buffers, force write of file
            else:
                print("Failed to process after multiple retry attempts.")
                f.write(f"{id}, Failed to process after multiple retry attempts.\n")
                f.flush()
                os.fsync(f.fileno())

def getIndex(id):
    index = hashtag_df.index[hashtag_df.id == id]
    return index[0]

def fillInMissingIds():
    generated_df = pd.read_csv("hashtag.csv", sep=',')
    missing_values = set(hashtag_df.iloc[:, 0]).symmetric_difference(set(generated_df.iloc[:, 0]))
    idxs = [getIndex(id) for id in missing_values]
    generate(idxs)
    return list(missing_values)

if __name__ == "__main__":
    # fillInMissingIds()
    generated_df = pd.read_csv("hashtag.csv", sep=',')
    print(f"original length: {len(hashtag_df['id'])}, generated length: {len(generated_df['id'])}")
    print(f"Are there any duplicates? : {generated_df.duplicated(subset=['id'], keep='last').any()}")