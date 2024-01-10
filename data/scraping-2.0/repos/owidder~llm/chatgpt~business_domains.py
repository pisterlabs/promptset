import os
import sys

import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


def process_one_file(in_filepath: str) -> bool:
  out_filepath = f"{'.'.join(in_filepath.split('.')[0:-1])}.bd"
  if os.path.exists(out_filepath):
    return False
  else:
    print(f"Creating: {out_filepath}")
    content = open(in_filepath, "r").read()
    print(content)
    words = []
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "You will be provided with a list of words, and your task is to detect the business domain each word belongs to. "
                     "Please do only answer with the names of the business domains. No other text!"
        },
        {
          "role": "user",
          "content": content
        }
      ],
      temperature=0,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    for i in range(len(response["choices"])):
      words.extend(response["choices"][i]["message"]["content"].split("\n"))
    with open(out_filepath, "w") as out:
      out_words = "\n".join(words)
      out.write(out_words)
      print(f"--> {out_words}")
    return True


def process_folder(folder_path: str):
  counter = 0
  for subdir, dirs, files in os.walk(folder_path):
      for file in files:
        if file.endswith("py._long_words_") and not file.endswith("__init__.py._long_words_"):
          file_abs_path = subdir + os.path.sep + file
          processed = process_one_file(file_abs_path)
          counter += 1
          if processed:
            print(counter)
            if counter > 1000:
              sys.exit(0)


process_folder("/Users/oliverwidder/dev/github/nlp/dict/_all/antlr/words")
