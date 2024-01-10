import openai
import time
import pandas as pd



APIKEY = open(r"C:\Users\beyou\Desktop\Token.txt","r")
inputQuery = open("Test.txt", "r")
result = open("result.txt", "a")

openai.api_key = APIKEY.readline()
query = inputQuery.readlines()

queryCount = len(query)
flag = 0
baseWait = 3
i = 0

while i < queryCount:
    if flag != i:
        baseWait = 3 
    try:
        completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        temperature = 0.2,
        max_tokens = 500,
        messages = [
            {"role": "system", "content": "Grade the following sample strictly following the 2019 DP IB english grading rubric, be sure to give a grade out of 5 for every criteria and keep the total word count below 200"},
            {"role": "user", "content": query[i]}
        ]   
        )
        
        result.write("Sample " + str(i + 1) + "\n" + str(completion.choices[0].message) + "\n")
        time.sleep(3)
    except:
        print("Error with server on sample " + str(i + 1))
        if flag == i:
            baseWait = baseWait + 2
            
        flag = i
        i-= 1
        
        if baseWait == 12:
            print("Timeout on sample " + str(i + 1))
            break
        time.sleep(baseWait)
        
        
    i+= 1
    
        

result.close()
APIKEY.close()
inputQuery.close()