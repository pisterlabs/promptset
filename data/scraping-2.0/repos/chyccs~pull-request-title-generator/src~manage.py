import inspect
import os

import openai

from main import fetch_pull_request


def _logging(level: str, title: str, message: str):
    frame: inspect.FrameInfo = inspect.stack()[2]
    print(f'::{level} title={title}::{message}, file={frame.filename}, line={frame.lineno}')


def _required(title: str):
    return 'fill me' in title.lower()


def main():
    openai.organization = os.getenv("open_ai_org", "org-JAnMEEEFNvtPSGwRA1DVF3Yu")
    openai.api_key = os.getenv("open_ai_api_key", "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")

    pull_request = fetch_pull_request(
        access_token=os.getenv("access_token"),
        owner=os.getenv("owner"),
        repository=os.getenv("repository"),
        number=int(os.getenv("pull_request_number")),
    )

    if not _required(pull_request.title):
        return

    patches = []

    for f in pull_request.get_files():
        patches.append(f'###### Modifications of {f.filename}')
        patches.append(f.patch)

    patches.append(
        '###### Please summarize this source code changes in one comprehensive sentence of 30 characters or less.'
        'Then modify it according to Conventional Commits 1.0.0 rules'
    )
    prompt = '\n'.join(patches)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["######"],
    )

    _logging(level='info', title=f'openai : {response}', message='')

    pull_request.edit(
        title=(response['choices'][0]['text']),
    )


if __name__ == "__main__":
    main()
