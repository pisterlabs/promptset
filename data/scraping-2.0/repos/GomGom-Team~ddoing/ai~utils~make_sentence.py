import json
import openai
import asyncio
import urllib.request

openai.api_key = ''
papago_id = ''
papago_secret = ''
papago_url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
# print(openai.Model.list())

#images
'''
command = dict(prompt="Mona Lisa", n=1, size='512x512')

temp = openai.Image.create(**command)

print(temp)
'''

async def make_sentences(word : str)->list:
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        n=10,
        messages = [
            {
                "role" : "user", 
                "content" : f"can you make sentences using {word} for kindergarten?"
            }
        ]
    )

    temp = completion.choices[0].message.content.split('\n')
    temp = {t[(idx//10)+3:].lstrip():'' for idx,t in enumerate(temp)}
    
    for key in temp:
        result = await translate(key)
        if result:
            temp[key] = result

    return temp

async def translate(sentence : str)->str:
    encode_txt = urllib.parse.quote(sentence)
    data = f'source=en&target=ko&text={encode_txt}'
    request = urllib.request.Request(papago_url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", papago_id)
    request.add_header("X-NCP-APIGW-API-KEY", papago_secret)

    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()

    if rescode == 200:
        result = response.read().decode('utf-8')
        result = json.loads(result)
        return result['message']['result']['translatedText']
    return None

#chat

if __name__ == '__main__':

    word = 'shoe'

    result = asyncio.run(make_sentences(word))
    # result = asyncio.run((translate("Hello world")))
    print(result)
    k_len, v_len = 0,0
    for k,v in result.items():
        k_len += len(k)
        v_len += len(v)

    print(f'result : eng = {k_len}, kor = {v_len}')
