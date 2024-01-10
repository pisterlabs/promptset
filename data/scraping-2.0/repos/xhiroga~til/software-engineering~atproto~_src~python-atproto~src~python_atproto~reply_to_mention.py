import os
from typing import Dict

import openai
from atproto import Client
from dotenv import load_dotenv

load_dotenv(verbose=True)

HANDLE = os.getenv('HANDLE')
PASSWORD = os.getenv('PASSWORD')
DID = os.getenv('DID')

openai.organization = os.environ.get("OPENAI_ORGANIZATION")
openai.api_key = os.environ.get("OPENAI_API_KEY")


def is_mention(feature: Dict[str, str], did: str) -> bool:
    return feature['_type'] == 'app.bsky.richtext.facet#mention' and feature['did'] == did


def generate_reply(text):
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": text}])
    first = chat_completion.choices[0]
    return first.message.content


def reply_to(post):
    parent = {
        "cid": post.cid,
        "uri": post.uri,
    }
    if post.record.reply is None:
        return {
            "root": parent, "parent": parent
        }
    else:
        return {
            "root": post.record.reply.root, "parent": parent
        }


def main():
    client = Client()
    client.login(HANDLE, PASSWORD)

    # TODO: 最後に取得した日付を指定する
    timeline = client.bsky.feed.get_timeline({'algorithm': 'reverse-chronological'})
    for feed_view in timeline.feed:
        facets = feed_view.post.record.facets
        if facets is not None and any([any([is_mention(feature, DID) for feature in facet.features]) for facet in facets]):
            reply = generate_reply(feed_view.post.record.text)

            client.send_post(text=f"{reply}", reply_to=reply_to(feed_view.post))


if __name__ == '__main__':
    main()
