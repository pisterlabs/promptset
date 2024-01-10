import os
import openai
import argparse
import sys

parser = argparse.ArgumentParser()
# parser.add_argument('-q')

path = os.getcwd()

print(sys.argv[2:])
file_path = f"{path}/api.txt"

openai.api_key = "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"
args = sys.argv[2:]
# args = parser.parse_args()
prompt = " ".join(args)
print(prompt)
# prompt = "넌 누가 만들었어"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)
# generated_text = response.choices[0].text.strip()
print(response["choices"][0])
results = response["choices"][0]["message"]["content"]
print("￦n")
print("______________________")

# print(generated_text)

# write txt file
f = open("ChatGPT_termux/result.txt", "w")
f.write(results)
