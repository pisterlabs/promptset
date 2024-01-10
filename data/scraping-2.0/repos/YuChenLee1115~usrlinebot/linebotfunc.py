# coding=utf8
from flask import Flask
app = Flask(__name__)

from flask import request, abort
from linebot import  LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage,TextSendMessage, ImageSendMessage, StickerSendMessage, LocationSendMessage, QuickReply, QuickReplyButton, MessageAction

import openai

line_bot_api = LineBotApi('3P2FZBfbBBPvoGl+gx6OA3sNrZzyFmA2d8GirkolO74hEyDAXbruL5iWGPKJ08aoZ15p/mvO6yyMjqcgFdQ+RpLzqWshDLt2W0LG38TCzwduHazxqG6+9dsBgstdfRPdJlK9+J0lAuxU14D/cK65ygdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('17ba2f1582e0779e70e2737380b5a6e9')

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
        url = 'https://usrlinebot.onrender.com/' + 'render_wake_up'
        res = requests.get(url)
        if res.status_code==200:
            print('喚醒render成功')
        else:
            print('喚醒失敗')
        time.sleep(12*60)

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

    elif mtext == '想要瞭解更多剪黏的問題' or mtext == "H" or mtext == "h":
        try:
            message = [
                TextSendMessage(  
                text = "以下有幾題關於剪黏的QA，歡迎來看看，請您使用圖文選單可以提供您快速選單或是請您回答題數，如:1\n\n1.什麼是剪黏？\n2.是剪黏「司阜」，還是「師傅」?\n3.現在還有人在使用碗或是玻璃剪黏嗎?\n4.古蹟修復完成後，會呈現出什麼樣的風格?\n5.剪黏司阜都在什麼樣的環境下工作?\n6.為何不等到作品完成後再運至屋頂即可?\n7.除了天氣不好外，白天黑夜都必須在上面工作嗎?\n8.人偶臉部的表情是怎麼製作出來的呢?\n9.廟裡為何會有水族動物的剪黏作品?\n10.什麼是「留灰縫」與「不見灰」?"
                ),
                TextSendMessage(
                text='請選擇您想要問的問題\n點擊以下的按鈕，會幫您快速提問'
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="什麼是剪黏", text="什麼是剪黏?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="是剪黏「司阜」，還是「師傅」", text="是剪黏「司阜」，還是「師傅」?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="現在還有人在使用碗或是玻璃剪黏嗎", text="現在還有人在使用碗或是玻璃剪黏嗎?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="古蹟修復完成後會呈現出什麼樣的風格", text="古蹟修復完成後會呈現出什麼樣的風格?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪黏司阜都在什麼樣的環境下工作", text="剪黏司阜都在什麼樣的環境下工作?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="為何不等到作品完成後再運至屋頂即可", text="為何不等到作品完成後再運至屋頂即可?")
                        ),                    
                        QuickReplyButton(
                            action=MessageAction(label="除了天氣不好外白天黑夜都必須在上面工作嗎", text="除了天氣不好外白天黑夜都必須在上面工作嗎?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="人偶臉部的表情是怎麼製作出來的呢", text="人偶臉部的表情是怎麼製作出來的呢?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="廟裡為何會有水族動物的剪黏作品", text="廟裡為何會有水族動物的剪黏作品?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="什麼是「留灰縫」與「不見灰」", text="什麼是「留灰縫」與「不見灰」?")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    
    elif mtext == '什麼是剪黏?' or mtext == "1":
        try:
            message = TextSendMessage(  
                text = "「剪黏技藝」是一個總稱，是中國南方的廟宇古厝等傳統建築中特有的工藝形式。剪黏在一百多年前流傳至臺灣後，經過彼此的合作與競爭，也發展出屬於臺灣獨有的風格。廣義的「剪黏技藝」，可運用於三種不同的材質與技法，分別為「剪黏」、「泥塑」、「交趾陶」，這三種工法對於廟宇建築而言各有其特色與重要性！"
            )
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '是剪黏「司阜」，還是「師傅」?' or mtext == "2":
        try:
            message = TextSendMessage(  
                text = "「司阜」這個詞彙是對於特殊傳統工藝匠司的尊稱。18因此除了剪黏司阜以外，也會有雕刻司阜或是木工司阜等等的尊稱存在，所以下次看到有人被稱為司阜，不要再問是不是寫錯字囉，而是一個可以給他滿滿尊敬的稱呼!"
            )
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '現在還有人在使用碗或是玻璃剪黏嗎?' or mtext == "3":
        try:
            message = [
                TextSendMessage(  #傳送文字
                    text = "玻璃由於保存壽命不長等緣故，除了特定廟宇修復指定外，已較少出現。碗片則是古蹟修復時，為了依照廟宇原先的製作工法及初始廟貌等目的而使用。這些傳統的剪黏方式都需要耗費大量的時間及材料成本，也考驗著司阜技巧。下圖是惠濟宮整修時拆卸下來的玻璃剪黏作品，作品身上玻璃幾近剝落，露出大面積的粗胚，臉上彩繪顏料也有褪色的跡象"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/Qbl9fhU.jpg",
                    preview_image_url = "https://i.imgur.com/Qbl9fhU.jpg"
                ),
                TextSendMessage(  #傳送文字
                    text = "下圖則是惠濟宮整修過程，《黃忠》碗片剪黏半成品。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/SXy7W0I.jpg",
                    preview_image_url = "https://i.imgur.com/SXy7W0I.jpg"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    
    elif mtext == '古蹟修復完成後會呈現出什麼樣的風格?' or mtext == "4":
        try:
            message = TextSendMessage(  
                text = "「修舊如舊、修舊如新」是廟宇古蹟修復時，兩種不同的做法。前者是修復完成後要讓古蹟呈現出歷經風霜與時代的舊貌，作品上可能還要特別處理讓其看起來有破舊的感覺；後者則是修復後要呈現出當初興建完成的模樣，也就是說即便是用碗瓷或玻璃剪黏，但宛如剛蓋好的新氣象，同時也會把修復的痕跡視為重要歷史而保留下來。19兩種方法各有特色，皆能再現古蹟風華。"
            )
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '剪黏司阜都在什麼樣的環境下工作?' or mtext == "5":
        try:
            message = [
                TextSendMessage(  #傳送文字
                    text = "我們會在哪邊看到剪黏作品，司阜就會在那個地方工作。因此廟頂就是剪黏司阜常常出現的工作環境，對他們來說，爬上爬下就像家常便飯。有時須把自己綁在屋脊上防止被風吹落，好像手裡的剪黏作品，一樣都被固定在屋脊上、一樣要站在上面忍受各種天氣、一樣都展現出活靈活現的姿態，也都一樣都是為了讓廟宇變得更漂亮、更有生命力而存在。圖片說明:陳威豪司阜於廟頂上施作「藻花」。司阜頭上除了有工地用的安全帽悶著，還需頂著南部酷熱的陽光，腳下也踩著兩三層樓高的鷹架，要有不怕熱、不怕高的勇氣，才有條件在廟頂上製作剪黏。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/fXbyond.jpg",
                    preview_image_url = "https://i.imgur.com/fXbyond.jpg"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '為何不等到作品完成後再運至屋頂即可?' or mtext == "6":
        try:
            message = TextSendMessage(  
                text = "屋頂上的作品，除了美觀之外，最重要的還是安全性的問題，要確保作品不能無故掉落，或是在遇到地震或強風時不穩，對於來廟裡參拜的信徒造成危險。除了較小型的作品可先另外製作，再固定於廟頂，較大型的例如龍鳳等主要物件，必須在粗胚階段時就要在廟頂製作，在打胚時將不銹鋼條(以前多使用銅條)嵌進廟頂的「脊」固定，這樣才能讓這些作品跟整個建築物接合在一起，較能抵擋颱風與地震之侵襲，使作品能安全固定於廟頂。"
            )
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '除了天氣不好外白天黑夜都必須在上面工作嗎?' or mtext == "7":
        try:
            message = [
                TextSendMessage(  #傳送文字
                    text = "在屋頂上施作時，必須配合陽光，有自然光司阜才會知道哪裡需要調整或加強，所以天黑之前，司阜們就會回到平地。有時進度落後或是時間不足時，晚上還會繼續在工地加班，做的就是白天在屋頂上施作作品時所需要的素材，例如龍的鱗片、人物的衣服或是其他部位需要的材料，隔天上工時，在把這些東西一併帶上屋頂。下圖說明屋頂上方的棚架地板上留下大量司阜修剪過後的碗片碎屑。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/9oP3LwJ.jpg",
                    preview_image_url = "https://i.imgur.com/9oP3LwJ.jpg"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '人偶臉部的表情是怎麼製作出來的呢?' or mtext == "8":
        try:
            message = [
                TextSendMessage(  #傳送文字
                    text = "主要會善用陶土燒製或是泥塑的方式來製作人物的臉。前者以陶土為材質，手工捏製出人物表情，由於陶土的可塑性較佳，可以描繪出表情細節較為細膩，後因大量使用之需求，亦會將自身派門風格的臉譜翻製成石膏模，以手工壓模的方式增加生產效率。20後者使用白灰泥或水泥為材質，完成後再使用顏料上色描繪出表情，表情與神情可因描繪手法有更多變化，另因泥塑之材質與剪黏本體相同，材質結合性較高，較不易有年久脫落之風險。\n下圖為1980年代作品《武將帶騎》，由陳威豪司阜提供。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/sQTp8Nk.jpg",
                    preview_image_url = "https://i.imgur.com/sQTp8Nk.jpg"
                ),
                TextSendMessage(  #傳送文字
                    text = "下圖則為惠濟宮整修時卸下之作品。兩者皆為玻璃剪黏，也出現大量斑駁的跡象，但人物的臉部由於使用陶土高溫燒製，表情與顏色較無褪色跡象。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/6IEjIdO.jpg?1",
                    preview_image_url = "https://i.imgur.com/6IEjIdO.jpg?1"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '廟裡為何會有水族動物的剪黏作品?' or mtext == "9":
        try:
            message = [
                TextSendMessage(  #傳送文字
                    text = "有些人物、動物或物件，之所以會出現在廟中，有其代表的涵義，有時是一種祝福、有時則是保佑建築物本身可以長長久久，讓藝術融合了更多的信仰風俗。廟頂上剪黏作品除龍鳳與人物之外，也常見各式水族，如:魚、蟹、蝦等，由於傳統廟宇多使用木造材質建構，最怕的就是火，也因此有這些象徵「水」元素的生物在廟上希望廟宇能避免祝融之災發生21。而常見於廟裡各個角落的鰲魚也有一樣的意涵，「鰲魚」由龍生九子之一的螭吻演變而來，龍頭魚身，傳說其屬水性，也能吞噬火災22，故常見於中國建築的各個角落，期許用火平安。\n下圖左為神農宮水族動物，而右為惠濟宮廟頂上的水族動物"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/etJQIP9.jpg",
                    preview_image_url = "https://i.imgur.com/etJQIP9.jpg"
                ),
                TextSendMessage(  #傳送文字
                    text = "下圖則為慈諴宮金爐以及惠濟宮廟頂垂帶鰲魚身影。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/x1OWz9x.jpg",
                    preview_image_url = "https://i.imgur.com/x1OWz9x.jpg"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '什麼是「留灰縫」與「不見灰」?' or mtext == "10":
        try:
            message = [
                TextSendMessage(  #傳送文字
                    text = "此為剪黏所使用之兩種不同風格之技法，以我們這次訪談到的陳威豪司阜為例，他所傳之工法即為「留灰縫」，也就是在各個瓷片之間預留一定距離的白灰縫，這樣的做法有幾項優點，其一，因瓷片為完整嵌入白灰泥中，可增加瓷片之耐久度，較不易因年久脫落，另一項優點，因剪黏作品與觀賞者常有一定之觀賞距離，預留之白灰邊，於遠處觀賞時可增加立體感，著名之洪坤福大師及派下之五虎將，如陳天乞司阜即使用留灰縫技法。而「不見灰」之技法為將瓷片完全覆蓋於白灰之上，其特色為瓷片剪製之手法非常細緻，如人物之戰甲（摃搥），或動物的絨毛，都能細膩的呈現，剪製之技巧非常高超，呈現作品的繁複之美，著名之何金龍大師及其派下之王保原司阜及使用不見灰技法。下圖為惠濟宮「文王拖車」剪黏作品，屬「留灰縫」工法。每個碗片之間皆留下一定的縫隙，就算遠看也能清楚其中的輪廓"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/mqeSlJS.jpg",
                    preview_image_url = "https://i.imgur.com/mqeSlJS.jpg"
                ),
                TextSendMessage(  #傳送文字
                    text = "而下圖則為旗津天后宮廟頂玻璃剪黏作品，屬「不留縫」工法。其中武將的戰甲（摃搥）與背旗都能更仔細的呈現，邊框部分則是司阜於玻璃上用顏料彩繪。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/6N69bvt.jpg",
                    preview_image_url = "https://i.imgur.com/6N69bvt.jpg1"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
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
                            action=MessageAction(label="B.為何計畫想要推廣剪黏", text="為何計畫想要推廣剪黏")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="C.請問士林三大廟是哪三間呢?", text="請問士林三大廟是哪三間呢?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="D.什麼是剪黏?", text="什麼是剪黏?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="E.請問剪黏常用的材質有哪些呢?", text="請問剪黏常用的材質有哪些呢?")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="F.請問剪黏會用到哪些工具呢?", text="請問剪黏會用到哪些工具呢?")
                        ),                    
                        QuickReplyButton(
                            action=MessageAction(label="G.我想看士林三大廟的精選剪黏作品介紹", text="我想看士林三大廟的精選剪黏作品介紹")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="H.想要瞭解更多剪黏的問題", text="想要瞭解更多剪黏的問題")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="I.製作團隊", text="製作團隊")
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
                    text = "東吳大學USR計畫「文化永續・城市創生：士林學之建構」，秉持大學社會責任實踐的精神，將大學中所學知識以及校園人才，用於社區，回饋在地。遂以士林地區為行動場域，蒐集在地文史資料、生態背景、地方特色等，並規劃相關走讀路線與培育導覽人才，期待能將這些士林文化傳遞給更多有興趣的人以及身在士林的東吳學子們。"
            ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/wPBWvsv.jpg",
                    preview_image_url = "https://imgur.com/wPBWvsv.jpg"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '為何計畫想要推廣剪黏' or mtext == "B" or mtext == "b":
        try:
            message = TextSendMessage(  
                text = "剪黏藝術是廟宇建築中不可或缺的一塊，從廟頂上的飛龍鳳凰、脊堵上的交趾陶再到廟門旁的龍虎堵，都是匠師聚精會神後的傑作，但平時去廟裡拜拜時，人們往往專注於祭祀的神明與所求的願望，忽略了周邊這些美麗的作品，在工業進步的發展下，許多剪黏藝品改用模具大量生產，傳統技藝也因此受到挑戰與凋零，希望剪黏板聊天機器人的存在，可以讓大家更加認識這個傳統技藝。"
            )
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '請問士林三大廟是哪三間呢?' or mtext == "C" or mtext == "c":
        try:
            message = [
                TextSendMessage(
                text='分別是舊街神農宮、芝山巖惠濟宮以及新街慈諴宮。請問想要先看哪一間廟的介紹呢 (請回答完整廟名)?'               
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="舊街神農宮", text="舊街神農宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖惠濟宮", text="芝山巖惠濟宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新街慈諴宮", text="新街慈諴宮")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '舊街神農宮':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "士林最古老的廟宇，前身「下樹林福德祠」，主祀福德正神，另陪祀先農等神明，始建於康熙48年(1709)年下樹林埔地(今新光醫院旁)，於乾隆六年(1741)因水災遭沖毀後，於舊街現址重建為「芝蘭廟」，並於嘉慶八年(1803)改祀神農大帝為主神，嘉慶九年改名「神農廟」，坐落舊街兩百餘年，見證舊街的貿易發展、漳泉械鬥等歷史，如今雖隱身巷弄，風華仍不減。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/dbVwtxs.jpg",
                    preview_image_url = "https://i.imgur.com/dbVwtxs.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="舊街神農宮", text="舊街神農宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖惠濟宮", text="芝山巖惠濟宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新街慈諴宮", text="新街慈諴宮")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '芝山巖惠濟宮':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "1752年於芝山岩初建，主祀神為開漳聖王，1751年黃澄清自漳州渡海來台時所攜帶之開漳聖王香火，後因靈驗才刻金身建廟祭拜。惠濟宮經過多次整建，將芝山寺的觀音佛祖與文昌祠的文昌帝君三廟合一，成為現今惠濟宮的樣貌，在清朝漳泉械鬥頻繁發生的背景下，開漳聖王成為士林、內湖一帶漳州先民的守護神，直至今日。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/1l2hqQq.jpg",
                    preview_image_url = "https://i.imgur.com/1l2hqQq.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="舊街神農宮", text="舊街神農宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖惠濟宮", text="芝山巖惠濟宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新街慈諴宮", text="新街慈諴宮")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '新街慈諴宮':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "1796年始建於芝蘭街，舊稱芝蘭街天后宮，主祀神為天上聖母。1859年因漳泉械鬥，天后宮與芝蘭街同遭焚毀，1864年方於新街重建，並於1875年的整修後正式命名為「慈諴宮」。媽祖廟建立之後，經歷了士林的繁華強盛、戰亂衰敗、重建、日治到現今人們心目中士林夜市的地標之一，是歷史的參與者和見證者。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/WCcjtqI.jpg",
                    preview_image_url = "https://i.imgur.com/WCcjtqI.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="舊街神農宮", text="舊街神農宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="芝山巖惠濟宮", text="芝山巖惠濟宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="新街慈諴宮", text="新街慈諴宮")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '什麼是剪黏' or mtext == "D" or mtext == "d":
        try:
            message = TextSendMessage(  
                text = "「剪黏技藝」是一個總稱，是中國南方的廟宇古厝等傳統建築中特有的工藝形式。剪黏在一百多年前流傳至臺灣後，經過彼此的合作與競爭，也發展出屬於臺灣獨有的風格。廣義的「剪黏技藝」，可運用於三種不同的材質與技法，分別為「剪黏」、「泥塑」、「交趾陶」，這三種工法對於廟宇建築而言各有其特色與重要性！"
            )
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '請問剪黏常用的材質有哪些呢?' or mtext == "E" or mtext == "e":
        try:
            message = [
                TextSendMessage(
                text='分別有碗片、玻璃以及淋搪。請問想先聽哪種材質的介紹呢 (請回答材質，如:碗片)?'               
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="碗片", text="碗片")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="玻璃", text="玻璃")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="淋搪", text="淋搪")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '碗片':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "碗片剪黏，由中國閩南一代傳入，以我們這次訪談的陳威豪司阜所師承的郭天來派剪黏為例，他們約莫在一百多年前來到臺灣，初期使用以前吃飯的碗、瓷器敲碎之後嵌於剪黏作品之上。現代由於古蹟修護的需求，古瓷碗已不易取得，則專門請陶瓷工廠製作剪黏專用的碗瓷。碗片與玻璃有個共同特色，那就是司阜可以利用碗及玻璃球原有的弧度鑲嵌於胚體上不同的部位，增加作品的立體感與變化性。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/9rLNZNV.jpg",
                    preview_image_url = "https://i.imgur.com/9rLNZNV.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="碗片", text="碗片")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="玻璃", text="玻璃")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="淋搪", text="淋搪")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    elif mtext == '玻璃':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "玻璃剪黏，大約在民國40-70年代間，隨著臺灣的工業發展，當時彩色玻璃燈罩大量出口，也被剪黏的匠師拿來嘗試作為剪黏素材，因材質能透光，在太陽光下非常璀璨耀眼顏色，也會因折射而有不同的美感，一時蔚為流行。由於當時需求量大，許多工廠會生產剪黏專用的玻璃球，使用前司阜再將其敲成片狀。但這樣的彩色玻璃有一致命缺陷，比起碗瓷，玻璃曝露在廟頂及戶外的空間，不耐熱漲冷縮，較難以承受過多的風吹日曬雨淋及溫度變化，壽命約十至十五年左右。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/rOrlp1v.jpg",
                    preview_image_url = "https://i.imgur.com/rOrlp1v.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="碗片", text="碗片")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="玻璃", text="玻璃")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="淋搪", text="淋搪")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))        
            
    elif mtext == '淋搪':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "淋搪剪黏，因為建廟的需求大量增加而產生的工法，將陶土漿澆灌在模具內，成型後燒製的陶瓷品。淋搪所生成的作品有大有小，可以直接製成一隻動物，或是龍爪、龍首等部位，也可以是各種形狀的馬賽克素材，省去了司阜將碗或玻璃剪成特定形狀的時間與人工，由於是高溫燒製，淋搪上面的顏色過三至五十年也不易褪色。有了固定的模具，除了大量生產之外，還能讓作品一體成形，縮短了每件作品的製程與成本，卻也讓剪黏手作的特色漸漸消失。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/IEunNTf.jpg",
                    preview_image_url = "https://i.imgur.com/IEunNTf.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="碗片", text="碗片")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="玻璃", text="玻璃")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="淋搪", text="淋搪")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '請問剪黏會用到哪些工具呢?' or mtext == "F" or mtext == "f":
        try:
            message = [
                TextSendMessage(
                text='剪黏常看見的工具有鑽筆、剪鉗、灰匙仔、土捧、不鏽鋼條、水泥等。請問想要先聽哪一種工具的介紹呢 (請回答工具名稱，如:鑽筆)?'               
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="鑽筆", text="鑽筆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪鉗", text="剪鉗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="灰匙仔", text="灰匙仔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土捧", text="土捧")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="不鏽鋼條", text="不鏽鋼條")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水泥", text="水泥")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '鑽筆':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "顧名思義，鑽筆的前緣裝有碎鑽，在剪黏的施作過程中，鑽筆主要是把碗公或是玻璃球初步切成適當的大小，後續再使用剪鉗將素材修剪成想要的形狀。切割時，鑽筆會在碗瓷或玻璃上留下痕跡，兩手再以那道痕跡為中心點向兩側掰開，即可分離。鑽筆切割時的力道，是學徒必須要仔細拿捏才能熟悉的技術。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/1pg8bba.png",
                    preview_image_url = "https://imgur.com/1pg8bba.png"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="鑽筆", text="鑽筆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪鉗", text="剪鉗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="灰匙仔", text="灰匙仔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土捧", text="土捧")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="不鏽鋼條", text="不鏽鋼條")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水泥", text="水泥")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '剪鉗':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "剪黏過程中，需要使用碗瓷或玻璃剪成大量且各種不同形狀的素材，有時是龍的鱗片、有時是動物的毛髮，還有許多人物所需要的戰甲或是武器，原材料用鑽筆初步切割後，就會用剪鉗細修成想要的形狀，雖然是剪，實際作法更接近用磨的方式磨出所需的形狀。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/TltNmEO.png",
                    preview_image_url = "https://imgur.com/TltNmEO.png"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="鑽筆", text="鑽筆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪鉗", text="剪鉗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="灰匙仔", text="灰匙仔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土捧", text="土捧")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="不鏽鋼條", text="不鏽鋼條")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水泥", text="水泥")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    
    elif mtext == '灰匙仔':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "主要用途有兩種，一是用來攪拌水泥，一是在粗胚過程中塑造形狀、細修紋路等。灰匙仔就像司阜手上的畫筆，在牆面或胚體上揮灑勾勒出栩栩如生的作品，灰匙仔的使用技巧是需要經過時間與經驗的累積才能運用自如。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/BJM1NlU.png",
                    preview_image_url = "https://imgur.com/BJM1NlU.png"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="鑽筆", text="鑽筆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪鉗", text="剪鉗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="灰匙仔", text="灰匙仔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土捧", text="土捧")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="不鏽鋼條", text="不鏽鋼條")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水泥", text="水泥")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '土捧':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "司阜於粗胚或施作泥塑時，用於承接水泥的器具，也能於上方持續翻攪水泥漿，增加水泥的彈性與緊實度。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/nRxFRUv.jpg",
                    preview_image_url = "https://i.imgur.com/nRxFRUv.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="鑽筆", text="鑽筆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪鉗", text="剪鉗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="灰匙仔", text="灰匙仔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土捧", text="土捧")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="不鏽鋼條", text="不鏽鋼條")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水泥", text="水泥")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '不鏽鋼條':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "不鏽鋼條是粗胚造型的重要骨架基底，司阜會依照作品須呈現的動作與架式調整不鏽鋼條的角度，再用白灰(水泥)一層層將其包覆。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/pnVF7Yv.jpg",
                    preview_image_url = "https://i.imgur.com/pnVF7Yv.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="鑽筆", text="鑽筆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪鉗", text="剪鉗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="灰匙仔", text="灰匙仔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土捧", text="土捧")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="不鏽鋼條", text="不鏽鋼條")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水泥", text="水泥")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '水泥':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "為剪黏作品之基底，傳統上使用白灰，現代因成本與便利性多使用水泥。作法是將水泥灰加水融合成水泥漿，並包覆在固定好架勢的不鏽鋼條上，層層堆疊塑造出作品的輪廓，於最後一層水泥未乾之時嵌入瓷片。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/pcueard.jpg",
                    preview_image_url = "https://i.imgur.com/pcueard.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="鑽筆", text="鑽筆")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="剪鉗", text="剪鉗")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="灰匙仔", text="灰匙仔")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="土捧", text="土捧")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="不鏽鋼條", text="不鏽鋼條")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="水泥", text="水泥")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

    elif mtext == '我想看士林三大廟的精選剪黏作品介紹' or mtext == "G" or mtext == "g":
        try:
            message = [
                TextSendMessage(
                text='請問想要先看神農宮、惠濟宮還是慈諴宮的精選作品介紹呢 (請回答廟名，如:神農宮)？'               
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="神農宮", text="神農宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="惠濟宮", text="惠濟宮")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="慈諴宮", text="慈諴宮")
                        )
                    ]
                )
            )
        ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '神農宮':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "以下將對於獅、渭水河、潼關遇馬超等作品介紹，想要先看哪一個作品的介紹呢?請回答作品名稱，如:獅"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="獅", text="獅")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="渭水河", text="渭水河")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="潼關遇馬超", text="潼關遇馬超")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                )
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    
    elif mtext == '獅':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "神農宮廟頂上的獅，為碗片剪黏之作品。與龍不同，龍身上有許多鱗片，多半是圓形或水滴型，身上的毛較少，而獅則全身幾乎都是由絨毛組成，司阜除了要準備許多形狀類似但長短不一或彎曲角度不同的絨毛狀碗片，也要使用不同的顏色來呈現出立體感與豐富度。獅臉部的各部位也是一大特點，如右圖可見獅的眼睛與唇邊皆是原本粗胚製作時預留下來的空間，並直接用顏料塗於胚體上，除了與眼珠上方的眉毛對比出層次外，眼神也更加深邃。司阜也利用不銹鋼絲呈現出臉上鬍鬚銀白、密集，且自然彎曲的特性。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/Cbdb8K8.jpg",
                    preview_image_url = "https://i.imgur.com/Cbdb8K8.jpg"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/BZebJEo.jpg",
                    preview_image_url = "https://i.imgur.com/BZebJEo.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="獅", text="獅")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="渭水河", text="渭水河")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="潼關遇馬超", text="潼關遇馬超")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '渭水河':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "「渭水河」為周文王至渭水河畔聘請姜子牙為丞相的故事。姜子牙因諫言得罪了商紂王，躲至渭水河畔隱居，並時常於此釣魚，其目的「不為錦鱗設，只釣王與侯」，也終於在老來八十之刻得到周文王的賞識，被封為丞相，並成功討伐暴政的紂王。這個故事也為籤詩中常見的典故，教人勿躁進，若做好準備，便等待機緣的到來。下圖作品位於神農宮正殿水車堵的位置，為釉上彩交趾陶作品。顏色鮮豔且對比明顯，作品中除了人物外，亦有用交趾陶製成的岩石、玻璃剪黏做的樹葉以及手繪的瀑布及天空遠景，其中岩石利用陶土塑形輔以釉料的配合，把石頭受到水或風侵蝕所造成的紋理呈現出來，讓岩石更立體，畫面也更富有層次感。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/ldrWKvC.jpg",
                    preview_image_url = "https://i.imgur.com/ldrWKvC.jpg"
                ),
                TextSendMessage(  #傳送文字
                    text = "下圖作品位於神農宮三川殿廟頂上牌頭之交趾陶作品，為近兩年廟宇整修時新塑。同為渭水聘賢的典故，也同為交趾陶工法製作，但由於呈現角度與製作年代的差異，創造出截然不同的風格。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/ph7STEa.jpg",
                    preview_image_url = "https://i.imgur.com/ph7STEa.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="獅", text="獅")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="渭水河", text="渭水河")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="潼關遇馬超", text="潼關遇馬超")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
    
    elif mtext == '潼關遇馬超':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "兩張圖分別為三川殿內樑堵間的交趾陶作品局部，以及廟頂牌頭整修後新作交趾陶作品，故事同為「潼關遇馬超」。此故事主要描述曹操欲攻下潼關，卻遭馬超反擊追殺的過程。下圖的作品很明顯可以看出馬超與曹操，兩位主角會比起其他兵將服裝更為鮮艷、豐富，有時也可用坐騎區分。另花臉為傳統戲曲中反派人物的常有特徵，又紅袍與長鬚皆為本故事中曹操的重要特徵，司阜用了這些典故製作出這個作品，讓大家在理解故事後更加清楚這些人物特色。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/qrjR2aG.jpg",
                    preview_image_url = "https://i.imgur.com/qrjR2aG.jpg"
                ),
                TextSendMessage(  #傳送文字
                    text = "下圖作品由於位於廟頂牌頭，空間的特性有別於前者，呈現出城外混戰的氣勢。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/FwVSfbn.jpg",
                    preview_image_url = "https://i.imgur.com/FwVSfbn.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="獅", text="獅")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="渭水河", text="渭水河")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="潼關遇馬超", text="潼關遇馬超")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '惠濟宮':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "以下將對於陳元光出征、四鳳亭、花瓶博古圖等作品介紹，想要先看哪一個作品的介紹呢 (請回答作品名稱，如:陳元光出征)?"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="陳元光出征", text="陳元光出征")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="四鳳亭", text="四鳳亭")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="花瓶博古圖", text="花瓶博古圖")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '陳元光出征':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "惠濟宮三川殿廟頂背面「陳元光出征」圖。陳元光為惠濟宮主神開漳聖王之本名，唐朝名將，曾出征平息寇亂等危機。作品中騎白馬的即是主角陳元光，旌旗上分別有「唐」、「陳」兩字，無論是馬身上的鬃毛、馬鞍、地勢上的高低起伏、花草植物，通通都用碗片製作，豐富作品的每一個層次，司阜也在碗片上繪製出人物的更多表情、戰甲上的細節等，讓整幅圖看起來更立體生動。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/uljxniI.jpg",
                    preview_image_url = "https://i.imgur.com/uljxniI.jpg"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/mMQl0uC.jpg",
                    preview_image_url = "https://i.imgur.com/mMQl0uC.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="陳元光出征", text="陳元光出征")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="四鳳亭", text="四鳳亭")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="花瓶博古圖", text="花瓶博古圖")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '四鳳亭':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "此作品為交趾陶，四鳳亭應為「四望亭」之誤寫。典故出自於清代《綠牡丹》第二十回「四望亭上女捉猴」。作品中四鳳亭下男性為駱宏勳，將救下在亭上因追逐白猴不慎摔落的花碧蓮。作品中人物的架式清楚且各有特色，表情也豐富多變，呈現出驚險一刻的緊張感，衣著顏色也與背景相互映襯，讓觀者知道主角是誰，而風景中的花草和柳葉，則使用彩色玻璃呈現，在光的映照下更加立體。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/Y88hn0J.jpg",
                    preview_image_url = "https://i.imgur.com/Y88hn0J.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="陳元光出征", text="陳元光出征")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="四鳳亭", text="四鳳亭")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="花瓶博古圖", text="花瓶博古圖")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '花瓶博古圖':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "此對花瓶博古圖位於惠濟宮二樓文昌閣左右兩側，為泥塑作品。文昌閣主祀文昌帝君，代表的是知識的追求以及文學的造詣，博古圖的風格正好與文人雅士的氣質相輔相成。與上一幅所介紹的博古圖不同，此對圖由於廟殿空間的限制，橫軸受到壓縮，遂以縱軸較長的特性放入花瓶與鮮花，泥塑打底並填上顏料，充分展現瓶身與花朵、樹葉的細節及紋路，也於瓶身繫上拂塵、如意等珍玩，將侷限的空間更有效發揮。"
                
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/g1rLrIj.jpg",
                    preview_image_url = "https://i.imgur.com/g1rLrIj.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="陳元光出征", text="陳元光出征")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="四鳳亭", text="四鳳亭")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="花瓶博古圖", text="花瓶博古圖")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '慈諴宮':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "以下將對於南極仙翁、八仙、屋頂上的武將等作品介紹，想要先看哪一個作品的介紹呢 (請回答作品名稱，如:南極仙翁)?"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="南極仙翁", text="南極仙翁")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="八仙", text="八仙")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="屋頂上的武將", text="屋頂上的武將")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '南極仙翁':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "慈諴宮的南極仙翁作品採用的是「碗片剪黏」的工法施作，除了南極真君以及童子兩位主要角色的頭是利用陶土燒製之外，無論是南極真君及身旁童子的衣服、白鶴、鳳、下方的祥雲、八仙堵上的人物以及牌頭上武將的戰甲、馬匹，通通都是用碗片一個個鑲嵌於粗胚之上，運用大量不同的碗片來呈現出每一位角色及物件上豐富的色調。顯現出剪黏司阜除了細緻的手法外，也需要具備用色的敏銳度，讓信眾從下方觀賞得以感受作品的立體感與特色。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/CQP8uB4.jpg",
                    preview_image_url = "https://i.imgur.com/CQP8uB4.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="南極仙翁", text="南極仙翁")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="八仙", text="八仙")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="屋頂上的武將", text="屋頂上的武將")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '八仙':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "八仙中除了人物各自的樣貌外，每位角色身上所持的器物、坐騎，也都有固定的搭配，也因此，在這方面的創作上就要極為細心，例如李鐵拐的虎，其黑色的花紋與黃色的皮膚，司阜用這兩種顏色的玻璃交錯鑲嵌，讓原先的玻璃片在觀眾的角度所呈現出來的是黑色的「條紋」，而何仙姑的鹿則是在上面附著大大小小的圓形黑白玻璃片，看起來就好像鹿本身的斑點，非常真實。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/IGDnPVc.jpg",
                    preview_image_url = "https://i.imgur.com/IGDnPVc.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="南極仙翁", text="南極仙翁")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="八仙", text="八仙")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="屋頂上的武將", text="屋頂上的武將")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == '屋頂上的武將':
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "位於屋頂上的武將相較於屋簷下的八仙圖，經過不斷的風吹雨淋以及陽光曝曬，有著非常明顯的碎裂以及剝落的現象，這樣的狀況也正如同前面介紹所說到玻璃的特性與缺點。目前依然能從圖片中看見這些仍嵌於胚體上的玻璃，在陽光的照射下透出更亮麗的色彩，這也是玻璃剪黏的美麗之處，而這些屹立不搖的胚體也告訴著大家，他們老了，卻依然生動魁梧。"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/uINhQb4.jpg",
                    preview_image_url = "https://i.imgur.com/uINhQb4.jpg"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://i.imgur.com/ZbELBCT.jpg",
                    preview_image_url = "https://i.imgur.com/ZbELBCT.jpg"
                ,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="南極仙翁", text="南極仙翁")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="八仙", text="八仙")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="屋頂上的武將", text="屋頂上的武將")
                        ),
                        QuickReplyButton(
                            action=MessageAction(label="回到上一層", text="g")
                        )
                    ]
                ))
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))
            
    elif mtext == 'I' or mtext == "i" or mtext == "製作團隊":
        try:
            message = [                
                TextSendMessage(  #傳送文字
                    text = "出  品：東吳大學USR計畫「文化永續・城市創生：士林學之建構」\n主  編：黃秀端 老師\n策  畫：陳威豪 助理\n指  導：陳威豪 司阜（永豐文化有限公司 負責人）"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/BU7Vpv5.jpg",
                    preview_image_url = "https://imgur.com/BU7Vpv5.jpg"
                ),
                ImageSendMessage(  #傳送圖片
                    original_content_url = "https://imgur.com/t6ab7uY.jpg",
                    preview_image_url = "https://imgur.com/t6ab7uY.jpg"
                )
            ]
            line_bot_api.reply_message(event.reply_token,message)
        except:
            line_bot_api.reply_message(event.reply_token,TextSendMessage(text='發生錯誤！'))

import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
