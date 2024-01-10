import requests
import argparse
from openai import OpenAI

def make_request(client, text, stream=False):
    response = client.chat.completions.create(
        model="test",
        messages=[
            {"role": "user", "content": text}
        ],
        stream=stream
    )
    if stream:
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                print(text, end='')
        print()
    else:
        print(response.choices[0].message.content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--stream', default=False, action='store_true')
    args = parser.parse_args()

    base_url = 'http://{}:{}/v1'.format(args.host, args.port)

    #base_url = 'http://192.168.139.143:8000/v1'
    # base_url = 'http://mc21.cs.purdue.edu:8000/v1'
    client = OpenAI(base_url=base_url, api_key="sk-xxx")

    while True:
        text = input(">>> ")
        make_request(client, text, stream=args.stream)




# content = "Compose a poem that explains the concept of recursion in programming."
# response = client.chat.completions.create(
#     model="test",
#     messages=[
#         {"role": "user", "content": content}
#     ],
# )
# print(response)
# print(response.choices[0].message.content)
