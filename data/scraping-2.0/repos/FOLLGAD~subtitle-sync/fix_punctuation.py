from openai import OpenAI

def fix_punctuation(text: str) -> str:
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
            "role": "system",
            "content": "Add punctuation to this text where it makes sense to improve the subtitles. DO NOT fix spelling or add/remove apostrophes. IMPORTANT: make sure to keep the order and all the characters the exact same. otherwise the software will break"
            },
            {
            "role": "user",
            "content": text,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content or ""
