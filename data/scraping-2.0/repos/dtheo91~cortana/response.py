import openai
import json
import sys

def jarvis():
    client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    data = sys.stdin.readlines()
    prompt = json.loads(data[0])

    response = client.chat.completions.create(
                                                model = "deepcoded/DeepCoder",
                                                messages = [{"role":"user", "content":prompt}],
    )

    sys.stdout.write(response.choices[0].message.content)
    sys.stdout.flush()

if __name__ == "__main__":
    jarvis()