import openai
import os
import json
import string
import time

openai.api_key = "..."

def get_detail(video_id):
    with open('liveoverflow_videos.jsonl', 'r') as f:
        for line in f.readlines():
            if video_id in line:
                print(line)
                return json.loads(line)


files = os.listdir("transcripts")
i = 0
for fname in files:
    transcript_file = f"transcripts/{fname}"
    video_id = fname.split('.')[0]
    i += 1
    print(f"{i}/{len(files)}")

    paragraphs = []
    with open(transcript_file, "r", encoding="utf-8") as f:
        text = f.read()
        paragraph = ""
        for line in text.splitlines():
            paragraph += line+" "
            if line.strip():
                if line[-1] in string.punctuation and len(paragraph) > 500 and len(paragraph) < 1500:
                    #print(paragraph)
                    details = get_detail(video_id)
                    prompt = f"""Video metadata:
Title: {details['title']}
Tags: {", ".join(details['tags'][:20]).strip()}
Description:
```
{details['description']}
```

Video transcript:
```
{paragraph.strip()}   
```

Write a technical question that can be answered by information in the video transcript:

"""
                    print(prompt)
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=300,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                        )
                    ai_data = response.choices[0].text
                    with open('500_metadata_finetune.jsonl', 'a') as f:
                        f.write(json.dumps({"prompt": ai_data.strip(), "completion": paragraph.strip()})+"\n")
                    paragraph = ""
                    print(ai_data)
                    time.sleep(1)
                    #input()
                    