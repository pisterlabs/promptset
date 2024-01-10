import json
import os
import openai
from tqdm import tqdm

with open("openai_key") as f:
    openai_key = f.read().strip()
    openai.api_key = openai_key

with open("all_scraped.json") as f:
    all_scraped = json.load(f)

RESULT_FILE_NAME = "chat_results.json"

if os.path.exists(RESULT_FILE_NAME):
    with open(RESULT_FILE_NAME) as f:
        results = json.load(f)
else:
    results = []

done_websites = []
for row in results:
    done_websites.append(row["website_name"])

MODEL = "gpt-3.5-turbo"

success = 0
failure = 0
for row in tqdm(all_scraped):
    website_name = row["website_name"]
    contents = row["contents"]

    if website_name in done_websites:
        continue

    try:
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Here is an website with URL {website_name}.\nHere are the contents of the website {contents}. What does this website do?",
            },
        ]
        result = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
        )
        description_response = result["choices"][0]["message"]
        # messages.append(description_response)
        # messages.append(
        #     {
        #         "role": "user",
        #         "content": "List 5 ideas how a large language model finetuned on question answering tasks can benefit their company.",
        #     }
        # )
        # result = openai.ChatCompletion.create(
        #     model=MODEL,
        #     messages=messages,
        # )
        # ideas_response = result["choices"][0]["message"]
        results.append(
            {
                "website_name": website_name,
                "description": description_response["content"],
                # "ideas": ideas_response["content"],
            }
        )
        with open(RESULT_FILE_NAME, "w") as f:
            json.dump(results, f, indent=2)
        success += 1
    except Exception as e:
        # print(website_name)
        # print(e)
        failure += 1

    with open("uptime", "w") as f:
        json.dump(
            {
                "failure": failure,
                "success": success,
            },
            f,
            indent=2,
        )
