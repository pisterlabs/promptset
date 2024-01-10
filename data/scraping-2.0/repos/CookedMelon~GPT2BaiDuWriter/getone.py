import openai
import os
openai.api_key = ''
noin=['简介', '课本', '答案', '通知书', '读本', '试卷']
def gettxt(title,temperature=0.8,max_tokens=120,top_p=1.0,frequency_penalty=0.5,presence_penalty=0.0):
  for i in noin:
    if i in title:
      print('不生成',title)
      return
  num=300
  if '作文' in title:
    num=800
  title2=title+'：'
  if num:
    title2=title+','+str(num)+'字：'
  article=title2
  while 1:
    response = openai.Completion.create(
      model='text-davinci-003',
      prompt=article,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
      stop=['生成结束'])
    if response.choices[0].text==''or len(response.choices[0].text)==1:
      break
    article+=response.choices[0].text
    print(response.choices[0].text)
    print(len(article))
    print('-------------------')
  path='./'+title+'.txt'
  print('write into',path)
  with open(path,'w',encoding='utf-8') as write_file:
    write_file.write(article[len(title2):].replace('\n','\n\n'))
  return article[len(title2):].replace('\n','\n\n')
  
gettxt('读《大众天文学》有感')