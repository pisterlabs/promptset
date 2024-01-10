import openai
openai.organization = "org-vWuBsijMsE4Lky7VKKbslmcG"
openai.api_key = "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"

def get_openai_response(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        n = 1,
        max_tokens=200
    )
    return completion.choices[0].message['content']