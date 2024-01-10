import os
import openai
import os
openai.organization = "org-NGeM7xd8YkcTE8569zCd785t"
openai.api_key = "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"
import tqdm
import time


def translate(code, path):
    prompt = f"""##### Translate this function from Java into C++
    ### Java

    {code}

    ### C++"""


    response = openai.Completion.create(
      model="code-davinci-002",
      prompt=prompt,
      temperature=0,
      max_tokens=1000,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["###"]
    )

    with open(path, "a+") as f2:
        for res in response["choices"]:
            f2.write(res["text"].strip())



dest = "/Users/drramos/PycharmProjects/SBFL_CPP/playground/java_codex"
files = os.listdir("./")
files = list(filter(lambda x: ".java" in x, files))
count = 0
for file in tqdm.tqdm(files):
    if count < 130:
        count += 1
        continue

    while True:
        try:
            src = os.path.join(os.getcwd(), file)
            with open(src, "r+") as f:
                code = f.read()
                translate(code, os.path.join(dest, file))
            break
        except:
            time.sleep(60)


