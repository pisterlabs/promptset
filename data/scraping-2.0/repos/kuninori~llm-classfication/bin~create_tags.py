import json

import openai

API_KEY = "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"
openai.api_key = API_KEY


def save_json():
    with open("./tags-maps.json", mode="w") as f:
        json.dump(d, f, ensure_ascii=False)


def create_tags(word, num):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"""
            あなたはニュース編集者です。
            """},
            {"role": "user", "content": f"""
            [{word}]というテキストについて{num}個、日本語のタグを返してください。
            結果をそのままJSONとして使用したいので不要な文字は含めないでください。
            JSONの構造は`[タグ1,タグ2]`の形としてください
            """}
        ]
    )
    tags_text = response["choices"][0]["message"]["content"]
    print(tags_text)
    return json.loads(tags_text)


d = {}
with open("./tags-maps.json", mode="r") as f:
    d = json.load(f)

words = [
    "天気",
    "災害",
    "地震",
    "台風",
    "社会",
    "経済",
    "科学",
    "文化",
    "季節",
    "政治",
    "国際",
    "ビジネス",
    "スポーツ",
    "暮らし",
    "地域",
    "人口",
    "世帯",
    "家計",
    "教育",
    "環境",
    "エネルギー",
    "企業",
    "司法",
    "IT",
    "自動車",
    "製造業",
    "AI",
    "半導体",
]

for w in words:
    if (w in d) == False:
        d[w] = create_tags(w, 10)
        print(w, d[w])
        save_json()
    for ww in d[w]:
        print(ww)
        if (ww in d) == False:
            tags = create_tags(ww, 5)
            d[ww] = tags
            print(ww, tags)
            save_json()

all = []
for k in d.keys():
    all.append(k)
    for t in d[k]:
        all.append(t)

with open("./tags.json", mode="w") as f:
    tags = list(set(all))
    json.dump(tags, f, ensure_ascii=False)
