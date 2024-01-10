import openai

question = input("Ask ChatGPT: ")

with open("message_history.txt","a") as myfile:
    myfile.write(question+"\n")
    
openai.api_key = open("key.txt","r").read() 
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}])

newTokens = int(completion["usage"]["total_tokens"])
currentTokens = int(open("tokens.txt","r").read())
newTokens = str(currentTokens+newTokens)
with open("tokens.txt","w") as myfile:
    myfile.write(newTokens)

content = completion["choices"][0]["message"]["content"]
print(content)
