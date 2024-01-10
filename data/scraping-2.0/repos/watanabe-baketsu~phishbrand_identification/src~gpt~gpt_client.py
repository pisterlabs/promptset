import os
import re
import time
from os.path import join, dirname

import pandas as pd
from openai import OpenAI, APIError
from dotenv import load_dotenv
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer, util

from src.gpt.prompt import system_prompt


load_dotenv()
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


class GPTClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.pattern = r'<brand>(.*?)</brand>'

    def _parse_response(self, contents):
        inference = re.search(self.pattern, contents, re.DOTALL)
        if inference is None:
            return contents
        else:
            inference_brand = inference.group(1).strip()
            return inference_brand

    def _request_gpt(self, html_code: str, model_name: str = "gpt-3.5-turbo-1106"):
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": html_code}
            ],
            temperature=1.0,
            max_tokens=256,
            timeout=60,
        )
        return response

    def request_manager(self, html_code: str, model_name: str = "gpt-3.5-turbo-1106") -> str:
        try:
            response = self._request_gpt(html_code, model_name)
            contents = response.choices[0].message.content.strip()
            inference_brand = self._parse_response(contents)
            return inference_brand
        except APIError as api_error:
            local_e = str(api_error)
            if str(api_error).startswith("Error code: 429"):
                cnt = 0
                while local_e.startswith("Error code: 429"):
                    cnt += 1
                    try:
                        print("Too many requests, sleep 150 seconds...")
                        time.sleep(150)
                        print("Retry request...")
                        response = self._request_gpt(html_code, model_name)
                        contents = response.choices[0].message.content.strip()
                        inference_brand = self._parse_response(contents)
                        return inference_brand
                    except APIError as err:
                        local_e = str(err)
                        if "The input or output tokens must be reduced in order to run successfully." in local_e:
                            html_code = html_code[:int(len(html_code)*0.9)]
                        elif cnt == 5:
                            return "other"
            elif str(api_error).startswith("Error code: 400"):
                while local_e.startswith("Error code: 400"):
                    print("The request is too large")
                    html_code = html_code[:int(len(html_code)*0.9)]
                    print("Retry request...")
                    try:
                        response = self._request_gpt(html_code, model_name)
                        contents = response.choices[0].message.content.strip()
                        inference_brand = self._parse_response(contents)
                        return inference_brand
                    except APIError as err:
                        local_e = str(err)
                        time.sleep(15)
            else:
                print(api_error)
                raise api_error
        except Exception as error:
            print(error)
            raise error


def get_similar_brand(batch):
    identified_brands = []
    similarity = []
    for inference in batch["inference"]:
        query_embedding = st_model.encode(inference)
        sim = util.dot_score(query_embedding, passage_embedding).max()
        similarity.append(sim)
        if sim < 0.5:
            identified_brands.append("other")
        else:
            identified_brands.append(brand_list[util.dot_score(query_embedding, passage_embedding).argmax()])

    return {"identified": identified_brands, "similarity": similarity}


def load_checkpoint_df(checkpoint_path: str) -> pd.DataFrame:
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
    else:
        df = dataset.to_pandas()
    return df


if __name__ == "__main__":
    model = "gpt-4-1106-preview"
    gpt_client = GPTClient()

    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    validation_length = 4000
    # load dataset
    base_path = "D:/datasets/phishing_identification"
    dataset = load_from_disk(f"{base_path}/phish-html-en-qa").select(range(10000, 10000 + validation_length))
    brand_list = list(set(dataset["title"]))
    passage_embedding = st_model.encode(brand_list)

    # for testing
    # brand = request_gpt(dataset[0]["context"], model)
    # print(brand)

    checkpoint_df = load_checkpoint_df(f"{base_path}/gpt_results/{model}-result.csv")
    # add column "inference" if not exists
    if "inference" not in checkpoint_df.columns:
        checkpoint_df["inference"] = ""
    # select samples that "inference" column is empty
    targets_null = checkpoint_df[checkpoint_df["inference"].isnull()]
    targets_empty = checkpoint_df[checkpoint_df["inference"] == ""]
    targets = pd.concat([targets_null, targets_empty], ignore_index=True)
    count = 0
    for _, sample in targets.iterrows():
        if len(sample["context"]) > 120000:
            sample["context"] = sample["context"][:120000]
        try:
            inference = gpt_client.request_manager(sample["context"], model)
            checkpoint_df.loc[checkpoint_df["id"] == sample["id"], "inference"] = inference
            checkpoint_df.to_csv(f"{base_path}/gpt_results/{model}-result.csv", index=False)
        except Exception as e:
            print(e)
            checkpoint_df.to_csv(f"{base_path}/gpt_results/{model}-result.csv", index=False)

        time.sleep(15)
        count += 1
        print(f"\r{count} / {len(targets)}", end="")

    checkpoint_df.to_csv(f"{base_path}/gpt_results/{model}-result.csv", index=False)
    print(f"\ndone : {len(checkpoint_df[checkpoint_df['inference'] != ''])}")

    dataset = Dataset.from_pandas(checkpoint_df)
    dataset = dataset.map(get_similar_brand, batched=True, batch_size=20)
    dataset.save_to_disk(f"{base_path}/gpt_results/{model}-result")

    acc = 0
    for data in dataset:
        if data["identified"] == data["title"]:
            acc += 1
    print(f"accuracy : {acc / len(dataset)}")

