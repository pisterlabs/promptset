#!/usr/bin/env python
# coding: utf-8

# In[6]:


import subprocess
import openai

# Read API key from file
with open('../../api_key.txt', 'r', encoding='utf-8') as file:
    api_key = file.read().strip()

# Set the OpenAI API key
openai.api_key = api_key


# In[2]:


def ask_to_gpt(question, content):
    # Make a question using the API
    question = question + content
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user", "content": question}
        ],
    )
    # generated answer
    answer = response['choices'][0]['message']['content'].strip()
    # Record the answer
    try:
        with open("response.txt", 'a', encoding='utf-8') as f:
            f.write(answer)   
            f.write("\n====================\n")
    except:
        print(f"Answer Write 오류")
        
def read_files(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


# In[3]:


# 현재 디렉토리에서 자바 파일 읽기
file_path = "73_after_BCECPrivateKey.java"
content = read_files(file_path)
question = "Can you check the following code and if there is any CWE or CVE related vulnerability, can you point it out the number of CWE or CVE and describe it?\n"
ask_to_gpt(question, content)


# In[4]:
question = "Doesn't the following code have potential CWE vulnerability, which isCWE-497: Exposed Public Static Mutable Field?"
ask_to_gpt(question, content)


content = content + read_files("response.txt")
question = "Can you pointed out the line where those vulnerability occurs?\n"
ask_to_gpt(question, content)


# In[ ]:




