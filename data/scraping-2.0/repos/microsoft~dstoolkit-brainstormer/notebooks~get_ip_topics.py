# %%
"""
This script generates one topic, keywords and topic description
for each ip based on ip fields configured. It is to provide
standard summarization of the ip and create more consistence
ip embedding
"""

import openai
import os
from pathlib import Path
import json
import time


openai.api_type = "azure"
openai.api_base = "https://jimyang-open-ai-001.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

ip_description_path = Path("../assets/ip_description.json")
ip_content_fields = ["ip_title", "scenarios", "business_value", "asset_description"]
# %%
with ip_description_path.open() as ip_file:
    ip_data = json.load(ip_file)
ip_data
# %%
ip_topic_prompts = []

for cur_ip in ip_data:
    ip_content = ""
    for field in ip_content_fields:
        if cur_ip[field] != "na":
            ip_content += f"""{field}: {cur_ip[field]}\
                """

    ip_topic_prompt = f"""
    You are given the overall description taken from a git-hub repos.
    Your task is to conduct following steps for a given description delimited by triple
    backticks:
    1. extract one most relevant data science topic from the description
    2. provide business user oriented explanation for the topic based on the description
    3. rewrite the summary limit the maximum length to 50 words
    4. extract top 5 data science keywords with business use case based on summary get 
    from step 2, separated the key words by comma
    

    ```{ip_content}```

    please format the topics and explanations as one JSON object with key of
    topic, explanation, short explaination, top keywords. Reply only with the JSON
    """

    ip_topic_prompts.append(ip_topic_prompt)


# %%
ip_topics_list = []
for ip_idx in range(len(ip_data)):
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a talented consulting data scientist. You will apply"
                "your knowledge of data science and machine learning to help a client "
                "to solve a business problem. "
            ),
        },
    ]
    conversation.append({"role": "user", "content": ip_topic_prompts[ip_idx]})

    ip_topic_response = openai.ChatCompletion.create(
        engine="ChatGPT",
        messages=conversation,
        temperature=0,
        max_tokens=4000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    cur_response = ip_topic_response["choices"][0]["message"]["content"]

    cur_ip_topic_dict = json.loads(cur_response)
    cur_ip_topic_dict["ip_id"] = ip_data[ip_idx]["id"]
    ip_topics_list.append(cur_ip_topic_dict)
    time.sleep(60)
# %%
# %%
topic_output_suffix = "-".join(ip_content_fields)
ip_topic_path = Path(f"../assets/ip_topics_{topic_output_suffix}.json")
with ip_topic_path.open("w") as fp:
    json.dump(ip_topics_list, fp)
