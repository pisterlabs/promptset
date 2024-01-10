import openai
import os
import time

openai.api_key = "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"
lang = ['python', 'c', 'c++', 'vb', 'swift', 'java', 'js', 'ruby', 'go', 'cs', 'r', 'powershell', 'lua', 'julia', 'PHP', 'Objective-C', 'vb.net7', 'html', 'coldfusion', 'cython', 'cpython', 'fortran', 'kotlin', 'matlab']

i = len(lang)
print(i)
f = open("output.txt", "w")

while(i>0):
    time_now = time.time()
    for n in range(0, len(lang)):
        prompt = ("write a code to print hello world uaing " + lang[n])
        completions = openai.Completion.create(engine="text-davinci-001", prompt=prompt, max_tokens=1280)
        message = completions.choices[0].text
        print(lang[n]+message+'\n', file=f)

    i-=i
    time_now = (time.time()-time_now)
    print('\n', "time spend:", time_now)

os.system('pause')