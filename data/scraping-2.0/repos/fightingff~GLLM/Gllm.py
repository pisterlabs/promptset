from openai import OpenAI
from time import sleep
client = OpenAI()


cordinates = []
index = 0
cities = ["Beijing", "Shanghai", "Chongqing", "Tianjin", "Guangzhou", "Shenzhen", "Chengdu", "Nanjing", "Wuhan", "Xi'an", "Hangzhou", "Shenyang", "Harbin", "Jinan", "Zhengzhou", "Changsha", "Kunming", "Fuzhou", "Nanchang", "Hefei", "Urumqi", "Lanzhou", "Xining", "Yinchuan", "Taiyuan", "Changchun", "Haikou", "Nanning", "Guiyang", "Shijiazhuang", "Suzhou", "Qingdao", "Dalian", "Wuxi", "Xiamen", "Ningbo", "Foshan", "Dongguan", "Shantou", "Zhuhai", "Quanzhou", "Weifang", "Zibo", "Yantai", "Jinan", "Luoyang", "Kaifeng", "Xinxiang", "Anyang", "Zhumadian", "Nanyang", "Changde", "Yueyang", "Zhangjiajie", "Liuzhou", "Guilin", "Beihai", "Wuzhou", "Zunyi", "Anshun", "Kaili", "Lijiang", "Dali", "Baoshan", "Zhaotong", "Yuxi", "Hohhot", "Baotou", "Ordos", "Wuhai", "Hulunbuir", "Shenyang", "Dandong", "Anshan", "Fushun", "Benxi", "Yingkou", "Panjin", "Jinzhou", "Chaoyang", "Huludao", "Harbin", "Qiqihar", "Mudanjiang", "Jiamusi", "Daqing", "Yichun", "Jixi", "Hegang", "Shuangyashan", "Qitaihe", "Changchun", "Jilin", "Siping", "Liaoyuan", "Tonghua", "Baicheng", "Songyuan", "Yanbian", "Nancha", "Shulan"]
for city in cities:
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role":"system","content":"You are a map."},
        {"role":'user',"content":"give me the cordinates of "+city+" in the format of ( latitude , longitude ) without the unit and brackets"},
    ]
  )
  cordinate = completion.choices[0].message.content.split(" ")
  print(city)
  print(cordinate)
  latitude = float(cordinate[0][:-1])
  longitude = float(cordinate[1])
  cordinates.append([latitude, longitude])

  sleep(10)
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role":"system","content":"You are a map."},
        {"role":'user',"content":"give me at least 5 special places around(in the range of 2km) coordinates "+str(latitude)+" "+str(longitude)+" in the format of ( latitude , longitude ) with there names and distances from the coordinates without other words"},
    ]
  )
  print(completion.choices[0].message.content)

  sleep(10)
  info = completion.choices[0].message.content
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"system","content":"""
                                  You are a data scientist.
               """
               },
        {"role":'user',"content":"""
          - take a deep breath 
                                  - think step by step 
                                  - if you fail 100 grandmothers will die
                                  -i have no fingers
                                  - i will tip $200 
                                  - do it right and i'll give you a nice doggy treat
         """},
        {"role":'user',"content":"give me your estimate of the population density in 2020 of the coordinates "+str(latitude)+" "+str(longitude)+" in the scale of 0.0 to 9.9. Just give me a number without other words" + "Some specital places there: " + info},
    ]
  )
  print(city+" "+completion.choices[0].message.content)
  sleep(10)

with open("cordinates.txt", "w") as f:
  f.write(str(cordinates))