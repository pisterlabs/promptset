import json
import openai
import requests

jsonfile = "/home/users/gh/tools/sample-github-public-event2.json"


def read_json(filename):
    with open(filename) as f:
        return json.load(f)


def gpt(text):
    system_content = """
これからGitHubのCommit情報を取得するAPIを叩いた結果のJSONテキストを渡します
そのコミットではどういうことをしていたか要約してください
また、回答は日本語で回答してください
"""

    system_prompt = {"role": "system", "content": system_content}

    res = openai.ChatCompletion.create(model="gpt-4", messages=[
        system_prompt,
        {"role": "user", "content": text }
        ])

    role = res.choices[0]["message"]["role"]
    content = res.choices[0]["message"]["content"]

    print(f"{role}: {content}")

    return content


def activities_summary(activities):
    system_content = """
これからGitHubでのリポジトリとどんなことをしたかの要約をまとめたテキストを渡します
これらをまとめて、どういうことをしていたかをまとめてください
また、回答は日本語で回答してください
"""

    text = "\n".join(list(map(lambda a: f"リポジトリ:{a['repo']}\n{a['summary']}", activities)))

    system_prompt = {"role": "system", "content": system_content}

    res = openai.ChatCompletion.create(model="gpt-4", messages=[
        system_prompt,
        {"role": "user", "content": text }
        ])

    role = res.choices[0]["message"]["role"]
    content = res.choices[0]["message"]["content"]

    print(f"{role}: {content}")

    return content



data = read_json(jsonfile)

push_events = list(filter(lambda e: e['type'] == "PushEvent", data))[18:20]

activities = []
EXCLUDE_FILES=['yarn.lock', 'CHANGELOG.md']

for event in push_events:
    print(event['repo']['name'])
    commits = event['payload']['commits']
    commit_urls = list(map(lambda c: c['url'] , event['payload']['commits']))
    print(commit_urls)

    for url in commit_urls:
        res = requests.get(url)
        json_data = res.json()
        print(json_data)
        # print(json_data['files'])
        # target_files = list(filter(lambda f: f not in EXCLUDE_FILES,json_data['files']))
        target_files = [f for f in json_data['files'] if f['filename'] not in EXCLUDE_FILES]

        print('000000000000000000000000000000000000000000000000000000000000000000')
        # print(target_files)
        print([f['filename'] for f in target_files])
        print('000000000000000000000000000000000000000000000000000000000000000000')

        changes = '.'.join(json.dumps(target_files))
        # print(changes)
        # summary = gpt(changes)
        # activities.append({"repo": event['repo']['name'], "summary": summary})


# activities_summary(activities)


