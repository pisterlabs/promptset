import os
import openai
import json
with open('secret.json', 'r') as file:
    data = json.load(file)
OPENAI_API_KEY = data.get("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
while True:  
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            #進入特殊模式後，需要另外寫進入該模式的邏輯
            {"role": "system", "content": "你是負責判斷句子中重點含義的人，需要注意的點有以下：當語意是「用AI繪圖」，則回應「AI生圖」；當語意是「自己拍的照片」，則回應「相簿取圖」；當語意是「幫我查詢...或是該句子可以透過網路得到答案」，則回應「爬蟲模式」；當語意是「使用簡報」，則回應「簡報模式」；當語意是「產生圖表」，則回應「圖表模式」；若語意沒有以上的內容，則回應「None」，不可以有其他類型的回應，僅可以回應以上我允許的文字"},
            {"role": "user", "content": "欸你看我昨天在日月潭拍的夕陽"},
            {"role": "assistant", "content": "相簿取圖"},
            {"role": "user", "content": "幫我用AI產生有一隻狗在草原上奔跑的圖"},
            {"role": "assistant", "content": "AI生圖"},
            {"role": "user", "content": "進入繪畫模式，然後幫我畫一個十字路口"},
            {"role": "assistant", "content": "畫布模式"},
            {"role": "user", "content": "進入圖表模式，五月的銷售額為20萬，六月的有90萬"},
            {"role": "assistant", "content": "圖表模式"},
            {"role": "user", "content": "好想睡覺"},
            {"role": "assistant", "content": "None"},
            {"role": "user", "content": "幫我查麥當勞優惠的新聞"},
            {"role": "assistant", "content": "爬蟲模式"},
            {"role": "user", "content": input("You: ")}
        ]
    )
    print(completion.choices[0].message.content)
