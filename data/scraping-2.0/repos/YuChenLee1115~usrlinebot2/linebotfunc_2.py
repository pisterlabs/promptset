# coding=utf8

from flask import Flask
app = Flask(__name__)

from flask import request, abort
from linebot import  LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage,TextSendMessage, ImageSendMessage, StickerSendMessage, LocationSendMessage, QuickReply, QuickReplyButton, MessageAction

import openai

line_bot_api = LineBotApi('KHGmE87M88v51uL4yvYp6Pk3rwjcIRb8W9zFcJCY7EDZDHVCqfJia8bdquHcH1DCRR2lNE2ZLv5DMVZEgiZtQtvM9uCiTgsPjwpx7zB9sjBtmjkb06rWE1aWnxqLdvtVhCHyOlcSzo0US4ENQeHfDAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('7c597d52fbcc02e219f288290f7e080b')

@app.route("/render_wake_up")
def render_wake_up():
    return "Hey!Wake Up!!"

#======python的函數庫==========
import time
#======python的函數庫==========

#======讓render不會睡著======
import threading 
import requests
def wake_up_render():
    while 1==1:
        url = 'https://usrlinebot2.onrender.com/' + 'render_wake_up'
        res = requests.get(url)
        if res.status_code==200:
            print('喚醒render成功')
        else:
            print('喚醒失敗')
        time.sleep(11.5*60)

threading.Thread(target=wake_up_render).start()
#======讓render不會睡著======

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    #ai_msg = event.message.text[:4].lower()
    mtext = event.message.text
    reply_msg = ''
    if mtext.startswith('gpt:'):
        try:
            openai.api_key = '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'
            # 將第六個字元之後的訊息發送給 OpenAI
            response = openai.Completion.create(
                        model='text-davinci-003',
                        prompt=event.message.text[4:],
                        max_tokens=256,
                        temperature=0.5,
                        )
            reply_msg = response["choices"][0]["text"].replace('\n','')
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text=reply_msg))
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == "常見問題":
        try:
            message = [
                TextSendMessage(
                text='請選擇您想要問的問題'
                ),
                TextSendMessage(
                text='點擊以下的按鈕，或是直接回覆英文字母，會幫您快速提問'
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="A.想請計畫自我介紹一下", text="想請計畫自我介紹一下")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="B.芝山巖慶讚中元的由來", text="芝山巖慶讚中元的由來")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="C.芝山巖四角頭簡介", text="芝山巖四角頭簡介")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="D.會前往哪幾間大墓公祭祀?", text="會前往哪幾間大墓公祭祀?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="E.想瞭解農曆七月的重要祭典儀式", text="想瞭解農曆七月的重要祭典儀式")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '想請計畫自我介紹一下' or mtext == "A" or mtext == "a":
        try:
            message = [
                TextSendMessage(  
                    text = "東吳大學USR計畫「文化永續‧城市創生：士林學之建構」，以士林宮廟、眷村、教育、生態、飲食等文化為始，並以「文史保存」與「在地人才培育」為主旨，期望能重現士林消失的集體記憶、打造士林文史工作人才培育機制，並有系統的建構屬於的士林文史資料庫。"
            ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/xvCbXxA.png",
                    preview_image_url = "https://imgur.com/xvCbXxA.png"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '芝山巖慶讚中元的由來' or mtext == "B" or mtext == "b":
        try:
            message = TextSendMessage(  
                text = "芝山巖大墓公最早是來自於清乾隆五十一年林爽文事件的死難者。清咸豐九年，漳泉械鬥後，亡故的先民也被共同收埋於芝山巖大墓公，地方為安撫亡魂、祈求合境平安，在農曆七月舉辦普度活動，流傳至今已逾一百六十多年，並在日治大正年間將士林北投漳州人開墾的聚落，分成四個角頭輪流值普。"
            )
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '芝山巖四角頭簡介' or mtext == "C" or mtext == "c":
        try:
            message = [
                TextSendMessage(
                text='四角頭分別為士林街角、石牌角、北山角、三芝蘭角，若以現代的行政區劃分，共有49個里屬於四角頭聯合普度的範圍，而每個角頭也有各自的主題特色。'               
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="士林街角", text="士林街角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="石牌角", text="石牌角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="北山角", text="北山角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="三芝蘭角", text="三芝蘭角")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '士林街角':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "士林街仔普電火，指的是新街居民會用成串的燈泡，照明裝飾擺設普品的供桌。範圍包含新街(慈諴宮、士林夜市)、舊街(神農宮、郭元益)等地。"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="士林街角", text="士林街角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="石牌角", text="石牌角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="北山角", text="北山角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="三芝蘭角", text="三芝蘭角")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '石牌角':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "石牌仔普紅龜粿，是因古早石牌種稻而祭拜紅龜粿等米製品。對照現今行政區大多座落於北投區，除天母里位於士林區。"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="士林街角", text="士林街角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="石牌角", text="石牌角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="北山角", text="北山角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="三芝蘭角", text="三芝蘭角")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '北山角':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "北山普豬公尾，是因過往陽明山上拜的豬公比較小隻。範圍約為陽明山一帶，包含內雙溪、外雙溪、山仔后(文化大學)、平等、菁礐仔等地。"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="士林街角", text="士林街角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="石牌角", text="石牌角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="北山角", text="北山角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="三芝蘭角", text="三芝蘭角")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '三芝蘭角':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "湳雅普傢伙，指的是蘭雅這邊有很多大地主準備豐盛的供品，傢伙又指家產。「三芝蘭」是三個地名合在一起的總稱，「三」代表三角埔(天母圓環、三玉宮一帶)，「芝」代表芝山巖，「蘭」代表蘭雅地區"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="士林街角", text="士林街角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="石牌角", text="石牌角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="北山角", text="北山角")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="三芝蘭角", text="三芝蘭角")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '會前往哪幾間大墓公祭祀?' or mtext == "D" or mtext == "d":
        try:
            message = [
                TextSendMessage(
                text='共會前往七間大墓公祭祀，分別為「芝山巖大墓公」、「牛踏橋保靈塔」、「水車邊萬善堂」、「林仔口萬善堂」、「永福里聖公媽廟」、「平等里坪頂萬善堂」、「內雙溪香對萬善堂」\n請問想要我先介紹哪間大墓公呢?'               
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '芝山巖大墓公':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "來自於清乾隆五十一年林爽文事件以及清咸豐九年漳泉械鬥士林地區死難的先人屍骨。漳泉械鬥後，地方開始祭拜這些無主孤魂安撫祂們，並祈求安定，成為芝山巖普度的起源。後來分成四個角頭輪流值普，大家更有充裕的時間及資金準備，四角頭聯合普度的傳統也延續至今。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/RY6mtOe.jpg",
                    preview_image_url = "https://imgur.com/RY6mtOe.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    elif mtext == '牛踏橋保靈塔':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "牛踏橋大約位在現今銘傳大學旁，與周圍山區屬於士林的公墓區，因銘傳建校遷移墳墓把當時無主或來不及遷移的墳墓集中整理在保靈塔。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/4Nkxqz3.jpg",
                    preview_image_url = "https://imgur.com/4Nkxqz3.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))        
            
    elif mtext == '林仔口萬善堂':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "位於林仔口，萬善堂內的墓碑上面寫著「塚避道路眾善同歸」，應為大正五年在開墾道路時挖到無主的骨骸，並集中收埋於此。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/cxMHFmd.jpg",
                    preview_image_url = "https://imgur.com/cxMHFmd.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '水車邊萬善堂':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "位於水車邊萬善堂旁，屋頂為圓頂狀，建築較新，也是收埋當地附近的屍骨，水車邊萬善堂的屍骨後來也一起被遷移至此祭祀。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/IIzmLW7.jpg",
                    preview_image_url = "https://imgur.com/IIzmLW7.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '永福里聖公媽廟':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "舊地名為拔子埔，內中的屍骨是在建設士林國中時挖出來的骨頭以及拔子埔在地的枯骨一起合葬。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/0spI5fk.jpg",
                    preview_image_url = "https://imgur.com/0spI5fk.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '平等里坪頂萬善堂':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "舊地名為坪頂，內中的屍骨主要是民國六十八年左右興建芝山公園時挖到的枯骨並集中在萬善堂祭祀。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/aADT4fo.jpg",
                    preview_image_url = "https://imgur.com/aADT4fo.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '內雙溪香對萬善堂':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "約有兩層樓高，前往的路途上須爬一小段山路，是民國七十一年時為了整建大度路挖到的屍骨並集中祭祀。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/2MWe0VZ.jpg",
                    preview_image_url = "https://imgur.com/2MWe0VZ.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖大墓公", text="芝山巖大墓公")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="牛踏橋保靈塔", text="牛踏橋保靈塔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水車邊萬善堂", text="水車邊萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="林仔口萬善堂", text="林仔口萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="永福里聖公媽廟", text="永福里聖公媽廟")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="平等里坪頂萬善堂", text="平等里坪頂萬善堂")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="內雙溪香對萬善堂", text="內雙溪香對萬善堂")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '想瞭解農曆七月的重要祭典儀式' or mtext == "E" or mtext == "e":
        try:
            message = [
                TextSendMessage(
                text='依照時間順序，有「開墓門」、「召請水陸孤魂儀式」、「中元普度」、「關墓門」等儀式，想先瞭解哪一個活動呢？'               
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="開墓門", text="開墓門")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="召請水陸孤魂儀式", text="召請水陸孤魂儀式")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="中元普度", text="中元普度")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="關墓門", text="關墓門")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '開墓門':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "農曆七月初一，釋教法師(俗稱黑頭司公)帶領隊伍前往四角頭境內的七間萬善堂開墓門。\n首先祭拜完土地公(后土)及大墓公後，法師唸經感謝土地神管理萬善堂的秩序。再與萬善堂的孤魂們說明現在已是農曆七月，可以自由活動，並通知祂們十四、十五日有相關的中元慶典可以來接受普度、饗食飽滿、聽法聞經，最後提醒祂們農曆八月初一關鬼門時，需準時回歸。\n儀式過程會搭配三牲、菜飯、餅乾等食物的祭拜供內中孤魂享用，也會燒化經衣、銀紙等紙錢給祂們使用。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/CpaL2AI.jpg",
                    preview_image_url = "https://imgur.com/CpaL2AI.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="開墓門", text="開墓門")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="召請水陸孤魂儀式", text="召請水陸孤魂儀式")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="中元普度", text="中元普度")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="關墓門", text="關墓門")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '召請水陸孤魂儀式':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "農曆七月十四日上午，法師會到芝山巖大墓公、惠濟宮旁石碑、神農宮石碑、牛踏橋保靈塔、林仔口萬善堂等地，邀請在這些陸地上遊蕩的孤魂回到芝山巖，準備接受隔日的普度。\n當晚，洲美碼頭邊有「放水燈」科儀，法師誦經完畢後，水燈頭將一個個被點亮，並推往水面，水中的孤魂也會隨著水燈頭串連的軌跡上岸，而岸邊的水燈，就像是街邊的路燈，照亮祂們前進的路。\n由水陸兩地召請來的孤魂，將在隔日芝山巖法會現場接受大家的普度。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/XIgdcy8.jpg",
                    preview_image_url = "https://imgur.com/XIgdcy8.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="開墓門", text="開墓門")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="召請水陸孤魂儀式", text="召請水陸孤魂儀式")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="中元普度", text="中元普度")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="關墓門", text="關墓門")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    
    elif mtext == '中元普度':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "農曆七月十五日上午恭祝地官大帝的誕辰，下午法師與祭拜隊伍將前往芝山巖大墓公、牛踏橋保靈塔、水車邊萬善堂、林仔口萬善堂，惠濟宮旁石碑、神農宮石碑等地以及當年值角的區域巡孤，再次召請與祭拜孤魂。\n在普度活動期間，孤魂來到芝山巖，就會暫時住在紙糊的「寒林院」、「同歸所」，法師會誦唸《梁皇寶懺》超度孤魂。普度當天除了孤棚裡的孤飯，還會準備豬公、牲禮、菜飯、蔬果、傳統糕點、糖果餅乾等豐盛的供品。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/2QEqsRf.jpg",
                    preview_image_url = "https://imgur.com/2QEqsRf.jpg"
                ),
                TextSendMessage(  #傳送文字
                    text = "當天還有三項重要科儀須舉行，請問您想先知道哪一個?"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="瑜珈焰口", text="瑜珈焰口")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="送大士爺", text="送大士爺")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="跳鍾馗", text="跳鍾馗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="e")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '瑜珈焰口':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "普度當天下午，會把大士山迎到孤棚上，晚上法師進行「放燄口」儀式，讓無法進食的孤魂，可以在這個時間緩解痛苦，順利享用普度的供品，供品當中的「佛手」及「包仔」，即是用來超度孤魂的最佳供品。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/lixnP9b.jpg",
                    preview_image_url = "https://imgur.com/lixnP9b.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="瑜珈焰口", text="瑜珈焰口")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="送大士爺", text="送大士爺")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="跳鍾馗", text="跳鍾馗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="e")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '送大士爺':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "「瑜珈焰口」科儀進行到「進包仔香」(召請孤魂的香)時，會將坐鎮於孤棚的面然大士(大士山)、寒林院、同歸所請至廟外燒化，請大士爺帶吃飽的孤魂離開人間。面然大士，觀音所化之憤怒貌，形象是青面獠牙、顏面被火燃燒的鬼王，維持普度會場的秩序。而土地公與山神，則是代表轄區的管理者，亦擔任守護道場的重責大任。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/7jGq98c.jpg",
                    preview_image_url = "https://imgur.com/7jGq98c.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="瑜珈焰口", text="瑜珈焰口")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="送大士爺", text="送大士爺")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="跳鍾馗", text="跳鍾馗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="e")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '跳鍾馗':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "普度活動結束後，由鍾馗跟孤魂們說明法事已經圓滿，並奉送祂們各回本位，若不從則加以驅趕，避免祂們徘徊，破壞秩序。儀式當中，無論是參與者或是旁觀者，都不能隨意呼叫他人姓名或大聲呼鬧，除了維持會場肅靜，也是避免人的魂魄被這些要離開的孤魂一起帶走。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/aZRh0Pk.jpg",
                    preview_image_url = "https://imgur.com/aZRh0Pk.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="瑜珈焰口", text="瑜珈焰口")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="送大士爺", text="送大士爺")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="跳鍾馗", text="跳鍾馗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="e")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == "關墓門":
        try:
            message = [
                TextSendMessage(
                text='農曆八月初一，法師前往七間萬善堂進行關門的科儀，向孤魂們說明農曆七月已經結束，請祂們在回到萬善堂後，保佑芝山巖四角頭合境平安，也請土地神們持續維護地方上的秩序。由於芝山巖開墓門的時間是七月初一午後，故於八月初一午後關墓門，給祂們完整一個月的自由時間。'               
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/S2h5jI6.jpg",
                    preview_image_url = "https://imgur.com/S2h5jI6.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="開墓門", text="開墓門")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="召請水陸孤魂儀式", text="召請水陸孤魂儀式")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="中元普度", text="中元普度")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="關墓門", text="關墓門")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
if __name__ == "__main__":
    app.run(debug = True , port = 8000)
