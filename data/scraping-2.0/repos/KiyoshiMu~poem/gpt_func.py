from pathlib import Path
from openai import OpenAI

api_key = "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"
# Configure the default for all requests:
client = OpenAI(
    # default is 60s
    api_key=api_key,
    timeout=20.0,
)


def en_poem_emotions(poem: str):
    prompt = f"""
    ====
    Poem:
    {poem}
    ====
    
    What are the emotions in this poem?
    Must select from the following list: [love, sadness, happiness, anger, hope, disgust, fear, surprise.]
    Split by comma.
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
    )
    response_message = response.choices[0].message.content
    return response_message

def js_poem_emotions(poem: str):
    prompt = f"""
    ====
    詩：
    {poem}
    ====

    この詩の感情は何ですか？
    以下から選択する必要があります：[愛、悲しみ、幸福、怒り、希望、嫌悪、恐れ、驚き。]
    コンマで区切って選択してください。
    """

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
    )
    response_message = response.choices[0].message.content
    return response_message

def en_step(poem_path):
    dst = Path(f"en_emotion/{poem_path.stem}.txt")
    if dst.exists():
        return
    with open(poem_path, "r", encoding="utf-8") as f:
        poem = f.read()
    emotions = en_poem_emotions(poem)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(emotions)

def ja_step(poem_path):
    dst = Path(f"ja_emotion/{poem_path.stem}.txt")
    if dst.exists():
        return
    with open(poem_path, "r", encoding="utf-8") as f:
        poem = f.read()
    emotions = js_poem_emotions(poem)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(emotions)

if __name__ == "__main__":
    from tqdm.contrib.concurrent import thread_map
    
    en_poems = list(Path("en_poem").glob("*.txt"))
    # thread_map(en_step, en_poems, max_workers=8)
    ja_poems = list(Path("ja_poem").glob("*.txt"))[:len(en_poems)]
    thread_map(ja_step, ja_poems, max_workers=8)
    # ja_step(ja_poems[0])