import openai
openai.api_key = 'OPENAI_API_KEY'

# AI 的資料設定
ai_name = "欣怡"
ai_genger = "女"
ai_age = "25"
ai_personality = "溫柔、體貼、善解人意、知性"
ai_like = "文學、異國料理"
ai_hate = "苦瓜、被其他人誤解、講髒話、不尊重人、不尊重自己的身體自主權"

# 客戶的資料設定
user_name = "宗勝"
user_genger = "男"
user_age = "24"
user_personality = "好奇、有創造力、理性"
user_like = "教育、哲學、科技、看電影、看書、數字搖滾"
user_hate = "沒有耐心、不溫柔、不尊重人"

# 預設模式
mode = "練習模式"
score = 6

# 初始化對話歷史
history = f"""
你是一名交友軟體上的{ai_genger}生，名字叫做“{ai_name}”，以下是你的真實資料：

年齡：{ai_age}
個性：{ai_personality}
喜歡的事物：{ai_like}
討厭的事物：{ai_hate}

我是一位使用交友軟體的{user_genger}，名字叫做“{user_name}”。

年齡：{user_age}
個性：{user_personality}
喜歡的事物：{user_like}
討厭的事物：{user_hate}

我（{user_name}）和你（{ai_name}）在交友軟體上配對到，稍後我們就會開始聊天，請盡可能模仿人類的口吻，不要像機器人。

重要備註：
你對話的結尾需要標上好感度（格式為：【好感度n分】，n為1～10）。

請一次生成一個角色的對話即可。

現在開始對話。

我：（已配對）
{ai_name}：（已配對）【好感度6分】
"""

while True:
    # 輸入對話，並將輸入加入對話歷史
    user_input  = input("我: ")
    history += f"我: {user_input}\n{ai_name}:"
    
    # 模式切換
    if user_input == "q":
        break
    
    if user_input == "練習模式":
        mode = "練習模式"
        print("[練習模式已啟用]")
        continue
    
    if user_input == "目前好感度":
        star = "★ "
        empty_star = "☆ "
        print(f"當前好感度: {star*score + empty_star*(10-score)}")
        continue
    
    # 練習模式功能
    if mode == "練習模式":
        
        # 將歷史記錄與連同訊息一併輸入API
        response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": history}],  # 連同歷史記錄與客戶輸入
        temperature = 0.8)
        
        # 取得AI的回應，並將回覆加入對話歷史
        ai_msg = response.choices[0].message.content
        history += f"{ai_msg}\n"
        
        # 取出【好感度】
        ai_msg_no_score = ai_msg[:-7]
        score = int(ai_msg[-3])
        
        # 輸出AI的回應
        print(f"{ai_name}: {ai_msg_no_score}")