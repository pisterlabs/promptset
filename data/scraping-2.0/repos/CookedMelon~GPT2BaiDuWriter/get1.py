import openai
import os
import pyttsx3
# openai.api_key = ''
def gettxt(title,type,api_key,temperature=0.8,max_tokens=120,top_p=1.0,frequency_penalty=0.5,presence_penalty=0.0):
  num=500
  openai.api_key=api_key
  if '作文' in title or '读后感' in title:
    num=800
  dir='./newarticles/'+str(type)
  filesexist=os.listdir(dir)
  for i in filesexist:
    if title in i:
      print('已存在',title)
      return
  title2=title+'：文中若出现电话号码或链接请打码'
  if num:
    title2=title+','+str(num)+'字：'
  article=title2
  errtime=0
  while 1:
    try:
      response = openai.Completion.create(
        model='text-davinci-003',
        prompt=article,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=['生成结束'])
      article+=response.choices[0].text
      print(response.choices[0].text)
      print(len(article))
      if response.choices[0].text==''or len(response.choices[0].text)==1:
        break
      print('-------------------')
    except Exception as e:
      errtime+=1
      print(e)
      # You exceeded your current quota, please check your plan and billing details.
      if errtime>3:
        engine = pyttsx3.init()  # 创建engine并初始化
        engine.say("遇到错误，程序已停止")
        engine.runAndWait()  # 等待语音播报完毕
        exit()
  path='./newarticles/'+str(type)+'/'+title+'.txt'
  print('write into',path)
  with open(path,'w',encoding='utf-8') as write_file:
    write_file.write('# '+title+article[len(title2):].replace('\n','\n\n'))
  return article[len(title2):].replace('\n','\n\n')