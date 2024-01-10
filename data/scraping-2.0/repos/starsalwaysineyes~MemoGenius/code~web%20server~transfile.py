import openai

filename=""
ifile=open("/www/wwwroot/mg.dawnaurora.top/test/input.txt",'r')
filename=ifile.read()
ifile.close()


openai.api_key ="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"
#"sk-enx8hSOzHczmdAxWpPHnT3BlbkFJ2vmUYNmWueckHruA1"
#"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"
#"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"

audio_file= open("/www/wwwroot/mg.dawnaurora.top/uploads/"+filename, "rb")

transcript = openai.Audio.transcribe("whisper-1", audio_file)

#print(transcript['text'])
with open("/www/wwwroot/mg.dawnaurora.top/test/transcribe_result.txt",'w') as f1:
    f1.write(transcript['text'])
    
cont="会议内容如下："+transcript['text']

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k-0613",
  messages=[
    {"role": "system", "content": "你是我的会议记录助手，你需要把会议中的内容都用中文进行要点总结，要能覆盖所有有意义的点。能按要点分层详细描述，必要时，要点可以超过30个，同时和数据有关的信息需全部记录为要点。必要时总结内容可以超过5000字"},
    {"role": "user", "content": cont}
  ],
  temperature=0.5
)

f=open("/www/wwwroot/mg.dawnaurora.top/test/result.txt",'w')
s=str(completion.choices[0].message["content"])#.decode("unicode-escape")
#print(s)
f.write(s)
f.close()





