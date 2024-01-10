import random
import json
import time
import sys
import re

import openai


PROMPT = \
"""### System
You are an AI Assistant. You must help the user to understand the following dialogue. While performing the task think step-by-step and justify your steps. Think like you are answering to a five-year-old.

### Dialogue
{dialogue}

### Human
This is a meeting log. I need to summarization which must contain main assertions and reasons for assertions. The summarization must be in Korean.
Hence, in order to summarize easily, you can write the assertions and their reasons in bullet points.
Also, before writing the summarization, you need to write the attendees of the meeting in bullet points.
Finally, you need to write a summary of the dialogue in 3 to 4 sentences, which are not in bullet points, based on the previous bullet points.
Therefore the format of the generated summarization will be like as below:

### 회의 참석자는 누구인가요?
- name1
- name2

### 주장을 하는 사람은 누구인가요?
- name

### 그 사람의 주장은 무엇이며, 그 이유와 근거는 무엇인가요?
- 주장: blah blah
- 근거: blah blah

### 요약:
blah blah x 3 ~ 4
"""

def get_augmentation(
    prompt,
    model_engine="text-davinci-003",
    max_tokens=2048,
    temperature=0.2,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
):
    if model_engine.startswith("text-davinci"):
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        return "\n".join(
            response["choices"][0]["text"].split("\\n")
        )
    elif model_engine.startswith("gpt-3.5-turbo"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content": prompt}
            ]
        )

        return "\n".join(
            response["choices"][0]["message"]["content"].split("\\n")
        )
    else:
        raise ValueError(f"model_engine {model_engine} is not supported")
    
def read_jsonl(fn):
    result = []
    with open(fn, "r") as f:
        for line in f:
            result.append(json.loads(line))
    return result


if __name__ == "__main__":
    input_fn = sys.argv[1]
    output_fn = sys.argv[2]
    api_key = sys.argv[3]
    start_index = int(sys.argv[4]) if len(sys.argv) > 5 else -1
    end_index = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    max_fail_cnt = 2

    openai.api_key = api_key

    n_generation = end_index - start_index if end_index > 0 else 2000

    jsonl = read_jsonl(input_fn)

    result = []

    iterator = iter(range(start_index, end_index)) if end_index > 0 else iter(random.sample(range(len(jsonl)), n_generation))

    fail_cnt = max_fail_cnt + 1
    while True:
        if fail_cnt > max_fail_cnt:
            idx = next(iterator)
            fail_cnt = 0

        js = jsonl[idx]

        try:
            utterances = js
            prompt = PROMPT.format(dialogue=utterances)
            augmentation = get_augmentation(
                prompt,
                model_engine="gpt-3.5-turbo",
            )

            print(augmentation + "\n\n")

            summary = {
                "summary": None,
                "attendees": None,
                "assertion_person": None,
                "assertion": None,
                "dialogue": utterances,
            }

            current_key = None
            for line in augmentation.split("\n"):
                if "회의 참석자는 누구인가요?" in line:
                    current_key = "attendees"
                    summary[current_key] = []
                    continue

                if "주장을 하는 사람은 누구인가요?" in line:
                    current_key = "assertion_person"
                    summary[current_key] = []
                    continue
                    
                if "그 사람의 주장은 무엇이며, 그 이유와 근거는 무엇인가요?" in line:
                    current_key = "assertion"
                    summary[current_key] = []
                    continue

                if "요약:" in line:
                    current_key = "summary"
                    summary[current_key] = []
                    continue

                if current_key is not None:
                    if line.strip().startswith("-"):
                        line = line.strip()[1:].strip()

                    if len(line.strip()) > 0:
                        summary[current_key].append(line.strip())

            if summary["summary"] is not None \
                and summary["assertion_person"] is not None and summary["assertion"] is not None \
                and summary["attendees"] is not None:

                print(utterances)
                print(json.dumps(summary, ensure_ascii=False, indent=4) + "\n\n")

                with open(output_fn, "a") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")

                fail_cnt = max_fail_cnt + 1
            else:
                fail_cnt += 1
        except Exception as e:
            print(e)
            time.sleep(45)
            fail_cnt += 1
