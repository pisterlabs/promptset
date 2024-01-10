import openai

import os
openai.api_key = os.environ.get("SECRET")


def getRequest(words,paragraph):
    startText = "Add "
    endText = " into the paragraph:"
    seperator = ", "
    joined = seperator.join(words)
    out = paragraph + "\n\n" + startText + joined + endText
    return out

def finishResume(words,paragraph):
    content = getRequest(words,paragraph)
    response = openai.Completion.create(model="text-davinci-002", prompt=content, temperature=0, max_tokens=min(100,max(len(paragraph)*2,20)))
    return response['choices'][0]['text'].strip()