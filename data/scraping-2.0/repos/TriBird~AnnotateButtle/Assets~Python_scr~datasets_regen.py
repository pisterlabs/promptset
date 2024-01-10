import openai
import pandas as pd
import json

with open('Assets\\Python_scr\\apikey.txt') as f:
    openai.api_key = f.read()

def ask():
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are an excellent planning professional."},
            {"role": "user", "content": f"""
                Please generate 10 30 character sentense on each of the following topics: sports, cooking, travel, languages, and music.
                The generated sentences will be treated as a machine learning dataset, so a variety of representations are required.
                Therefore, As three constraints, please avoid generating similar syntax and do not duplicate words used in the generated short sentences.
                Generated sentenses should not be quoted.
            """}
        ]
    )

    answer = response.choices[0]["message"]["content"]
    return answer

query_result = ask().replace('\n', '').split('.')

print(query_result)

datas = []
for q in query_result:
    if len(q) > 15:
        datas.append(q)

if not len(datas) == 50:
    raise ValueError("datasets must be 50. [", len(datas) ,"]")

datas_fordf = []
for i, d in enumerate(datas):
    if d[0] == " ":
        d = d[1:] + "."
    else:
        d = d + "."
    e = openai.Embedding.create(input=d, engine="text-embedding-ada-002")
    datas_fordf.append([(i // 10), d, e["data"][0]["embedding"]])
    print(i)

datas_df = pd.DataFrame(datas_fordf, columns=["target", "text", "embed"])

print(datas_df)

path = 'Assets\\Resources\\datasets_auto.json'
json_str = "{\"datasets\":" + datas_df.to_json(orient="records") + "}"

with open(path, mode="w") as f:
    f.write(json_str)