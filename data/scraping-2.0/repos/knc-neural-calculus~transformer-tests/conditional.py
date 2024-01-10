import os
import openai
import numpy as np

openai.api_key = os.environ["OPENAI_API_KEY"]


def filter_logprob(prompt, filter):
    combined = prompt + filter
    response = openai.Completion.create(
        engine="ada",
        prompt=combined,
        temperature=0.7,
        max_tokens=0,
        echo=True,
        top_p=1,
        n=1,
        logprobs=0
    )

    positions = response.choices[0]["logprobs"]["text_offset"]
    logprobs = response.choices[0]["logprobs"]["token_logprobs"]
    # tokens = response.choices[0]["logprobs"]["tokens"]

    word_index = positions.index(len(prompt))

    total_conditional_logprob = sum(logprobs[word_index:])

    return total_conditional_logprob


def filter_top_probs(preprompt, content, filter, quiet=0):
    index = 0
    logprobs = []
    substrings = []
    for word in content.split():
        index += len(word) + 1
        substring = content[:(index - 1)]
        prompt = preprompt + substring
        logprob = filter_logprob(prompt, filter)
        logprobs.append(logprob)
        substrings.append(substring)
        if not quiet:
            print(substring)
            print('logprob: ', logprob)

    return substrings, logprobs


def n_top_logprobs(preprompt, content, filter, n=5, quiet=0):
    substrings, logprobs = filter_top_probs(preprompt, content, filter, quiet)
    sorted_logprobs = np.argsort(logprobs)
    top = []
    for i in range(n):
        top.append({'substring': substrings[sorted_logprobs[-(i + 1)]],
                    'logprob': logprobs[sorted_logprobs[-(i + 1)]]})

    return top

f = open("preprompt.txt", "r")
preprompt = f.read()[:-1]
g = open("content.txt", "r")
content = g.read()[:-1]
h = open("filter.txt", "r")
filter = h.read()[:-1]

print('preprompt\n', preprompt)
print('\ncontent\n', content)
print('\nfilter\n', filter)

top = n_top_logprobs(preprompt, content, filter, 10)
print(top)

for t in top:
    print('\ncutoff: ', t['substring'][-100:])
    print('logprob: ', t['logprob'])
