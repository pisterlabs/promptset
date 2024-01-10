import openai
openai.api_key = "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"
NUM = 10


with open('ref.txt','r') as f:
    data = f.readlines()

datadict ={}
keys = []
values = []
qdata = []


for sen in data:
    sen = sen.strip()
    if(len(sen)!=0 and sen[-1]==":"):
        keys.append(sen[:-1])
        if(len(values)!=0):
            qdata.append(values)
        values=[]
    else:
        if(len(sen)!=0):
             values.append(sen.strip())
qdata.append(values)


for i in range(len(keys)):
    datadict[keys[i]]=qdata[i]

for key in datadict:
    for query in datadict[key]:
        response = openai.ChatCompletion.create(
                        model = "gpt-3.5-turbo",
                        messages = [{"role": "user", 
                            "content": f"Consider yourself a customer of a {key} , can you generate {NUM} queries that are similar to '{query}' "
                            }] )
        print(response)

        result = ''
        for choice in response.choices:
            result += choice.message.content

        with open('dataset.txt','a') as f:
            f.write('\n\n')
            f.write(query)
            f.write(result)