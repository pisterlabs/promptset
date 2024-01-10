from openai import OpenAI
from tqdm import tqdm
import json


def main():
    with open("data/test_zero.json") as file:
        data_list = json.load(file)
    api_key = "APIKEY"
    client = OpenAI(api_key=api_key)
    for data in tqdm(data_list["data"]):
        system_content = data["instruction"][:53]
        user_content = data["instruction"][53:]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            # model = "gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=200,
        )
        output = response.choices[0].message.content
        data["prediction"] = output[3:]

    json.dump(data_list["data"], open("GPT4_0.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
