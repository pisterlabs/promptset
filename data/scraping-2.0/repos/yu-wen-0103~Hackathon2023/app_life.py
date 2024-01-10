import os
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
from sql_db import EventDatabase
from pinecone_db import Pinecone_DB
import uuid
from langchain.chat_models import ChatOpenAI
from stage_select_tool import StageSelectionTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts.prompt import PromptTemplate



app = Flask(__name__)

os.environ["PINECONE_ENV"] = "gcp-starter"
os.environ["PINECONE_API_KEY"] = "6f605f74-d8ae-45c0-bdb1-aaf67686082b"
os.environ["OPENAI_API_KEY"] = "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"
channel_secret = '7f464ef8d999aae8a1bc7d18236fb5d9'
channel_access_token = 'q/FjCjRikjMrirNbJL0be4lI+6+a2ijAxJpq4NiNOSwwDC+Cw1mzCq6yLsSHu8vIR3o5dt61y8EseYffvlvud+U7PBwZeCeafM/TmoUdk6SP7jZQSiy2qCZ4EwAfYZsDTfi2HoZRmLf/uLFwLNWKjgdB04t89/1O/w1cDnyilFU='

# Channel Access Token
line_bot_api = LineBotApi(channel_access_token)
# Channel Secret
handler = WebhookHandler(channel_secret)


# 監聽所有來自 /callback 的 Post Request
@app.route("/webhooks/line", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


class ActionChooser():
    def __init__(self, db_name, table_name,  user_table_name):
        self.db_name = db_name
        self.table_name = table_name
        self.user_table_name = user_table_name
        self.mapping_stage_and_function_dict = {
            "我想選擇生活": 0, 
            "我想揪團": 2,  
            "主揪": 3, 
            "被揪": 4, 
            "查看我的活動": 5, 
            "我想去學習": 6,
            "活動序號:": 7,  # This is for the activity ID input
            "刪除": 8,  # This is for the delete activity input
            "想看看其他活動": 9,
            "揪團Deadline": 10,
            "結束問答": -1,
            "當期課程": 11,
            "課程名稱": 12,
            "查看作業": 13,
            "QA": 14,  
            "課堂討論": 15,
            "課程聊天室": 16,
            "課堂紀錄": 17,
            "正在產生彙整紀錄": 18,
            "進入失物招領": 19,
            "我想要張貼東西": 20,
            "物品名稱": 21,
            "我想要找東西": 22,
            "查物品pinecone": 23,
            "這是我的東西:" : 24,
            "作業問答中": 25,
        }

        # Database
        self.event_db = EventDatabase(self.db_name, self.table_name, user_table_name=self.user_table_name)
        self.Pinecone_DB = Pinecone_DB(self.db_name)
        
        # Langchain agent
        self.model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
        self.tools = [
            StageSelectionTool(),
        ]
        PREFIX = "You are an System Stage Selector. You have to choose a proper stage for the system according to the user's query. \
                  Use Stage Selection Tool to help you fullfill this job."
        SUFFIX = "Query: {input} (Please choose a stage for me, and only reply me a integer.)"
        self.stage_selection_agent = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            agent_kwargs={
                'prefix': PREFIX, 
                'suffix': SUFFIX
            },
        )
        self.qa_agent = ConversationChain(
            llm=self.model,
        )
    
    def run_chooser(self, event):
        text = event.message.text
        user_id = event.source.user_id
        user_stage_before = self.event_db.get_user_stage(user_id)
        print('text', text)
        #stage = self.substring_detector(text)
        
        # 我想要找東西
        if user_stage_before == 22:
            self.event_db.set_user_stage(user_id, 23)
            stage = 23
            return self.state_message_send(stage, user_id, text)
        
        if text == "結束問答":
            stage = -1
            self.event_db.set_user_stage(user_id, stage)
            return self.state_message_send(stage, user_id, text)

        if user_stage_before == 14:
            self.event_db.set_user_stage(user_id, 25)
            stage = 25
            return self.state_message_send(stage, user_id, text)

        if user_stage_before == 25:
            self.stage_selection_agent(user_id+' '+text)
            stage = self.event_db.get_user_stage(user_id)
            self.event_db.set_user_stage(user_id, stage)

            return self.state_message_send(stage, user_id, text)

        else:
            stage = self.substring_detector(text)
            if stage is None:
                self.stage_selection_agent(user_id+' '+text)
                stage = self.event_db.get_user_stage(user_id)
            self.event_db.set_user_stage(user_id, stage)
            return self.state_message_send(stage, user_id, text)
        
    def substring_detector(self, text):
        """
        Assume the the keys in the dictionary will be the substring of the text
        """
        for key in self.mapping_stage_and_function_dict.keys():
            if key in text:
                return self.mapping_stage_and_function_dict[key]
    
    def state_message_send(self, stage, user_id, text):
        if stage == 0:
            message = TemplateSendMessage(
                alt_text='Confirm template',
                template=ConfirmTemplate(
                    text='您需要什麼服務',
                    actions=[
                        MessageAction(
                            label='失物招領',
                            text='進入失物招領'
                        ),
                        MessageAction(
                            label='揪團活動',
                            text='我想揪團'
                        )
                    ]
                )
            )
            return message
        if stage == 1:
            raise NotImplementedError
        if stage == 2:
            message = TemplateSendMessage(
                alt_text='Buttons template',
                template=ButtonsTemplate(
                    # thumbnail_image_url='https://www.google.com/url?sa=i&url=https%3A%2F%2Fjome17.com%2F&psig=AOvVaw2yQ7E8QjPrYZ16KEJCGPld&ust=1697792244088000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCIiA2MH2gYIDFQAAAAAdAAAAABAI',
                    title='揪團囉~~',
                    text='請選擇想要的選項',
                    actions=[
                        MessageTemplateAction(
                            label='發起活動',
                            text='主揪'
                        ),
                        MessageTemplateAction(
                            label='現有活動',
                            text='被揪'
                        ),
                        MessageTemplateAction(
                            label='我的活動',
                            text='查看我的活動'
                        )
                    ]
                )
            )
            return message
        if stage == 3:
            message = TextSendMessage(text='請輸入以下資訊：\n1.揪團名稱:[請填入]\n2.揪團內容:[請填入]\n3.揪團時間:[請填入]\n4.揪團人數:[請填入]\n5.揪團地點:[請填入]\n6.揪團Deadline:[請填入]')
            return message
        if stage == 4:
            all_carousel_column = []
            event_top = self.event_db.get_events_sorted_by_deadline()
            print(event_top)
            print('event_top', event_top[0]['name'])
            for i in range(len(event_top)):
                carousel_column = CarouselColumn(
                    # thumbnail_image_url='https://example.com/item1.jpg',
                    title=str(event_top[i]['name']),
                    text='內容:'+ str(event_top[i]['describe']) + '\n時間:' + str(event_top[i]['time']) + '\n人數:' + str(event_top[i]['people']) +'\n地點:' + str(event_top[i]['location']) +'\n期限:' + str(event_top[i]['deadline']),
                    actions=[
                        MessageTemplateAction(
                            label='我要加入',
                            text='活動序號:' + str(event_top[i]['event_id'])
                        )
                    ]
                )
                all_carousel_column.append(carousel_column)
            
            all_carousel_column.append(
                CarouselColumn(
                    # thumbnail_image_url='https://example.com/item2.jpg',
                    title='發起活動',
                    text='我想要發起新活動',
                    actions=[
                        MessageTemplateAction(
                            label='發起活動',
                            text='主揪'
                        )
                    ]
                )
            )
            all_carousel_column.append(
                CarouselColumn(
                    # thumbnail_image_url='https://example.com/item2.jpg',
                    title='搜尋活動',
                    text='告訴我你想要的活動',
                    actions=[
                        MessageTemplateAction(
                            label='告訴我想要的活動',
                            text='想看看其他活動'
                        )
                    ]
                )
            )
            
            message = TemplateSendMessage(
                alt_text='Carousel template',
                template=CarouselTemplate(
                    columns=all_carousel_column
                )
            )
            return message
        if stage == 5:
            all_carousel_column = []
            user_events = self.event_db.print_events_for_user(user_id)
            for i in range(len(user_events)):
                carousel_column = CarouselColumn(
                    # thumbnail_image_url='https://example.com/item1.jpg',
                    title=user_events[i]['name'],
                    text='內容:'+ str(user_events[i]['describe']) + '\n時間:' + str(user_events[i]['time']) + '\n人數:' + str(user_events[i]['people']) +'\n地點:' + str(user_events[i]['location']) +'\n期限:' + str(user_events[i]['deadline']) ,
                    actions=[
                        MessageTemplateAction(
                            label='我要取消',
                            text='刪除' + str(user_events[i]['event_id'])
                        )
                    ]
                )
                all_carousel_column.append(carousel_column)
            all_carousel_column.append(
                CarouselColumn(
                    # thumbnail_image_url='https://example.com/item2.jpg',
                    title='課程區',
                    text='還敢參加那麼多活動阿\n你的教授is watching you!!',
                    actions=[
                        MessageTemplateAction(
                            label='我想去學習',
                            text='我想去學習'
                        )
                    ]
                )
            )
            message = TemplateSendMessage(
                alt_text='Carousel template',
                template=CarouselTemplate(
                    columns=all_carousel_column
                )
            )
            return message
        if stage == 6:
            message = TextSendMessage(text='請按一下下方menu的課程區')
            return message
        if stage == 7:
            event_join_db = {}
            event_join_db['user_id'] = user_id
            event_join_db['event_id'] = text[text.find('活動序號:')+5:]
            event_join = self.event_db.update_event_participation(event_join_db['event_id'], event_join_db['user_id'])
            
            #event_join = {} # from db
            message = TextSendMessage(
                text = '您已加入 '+ str(event_join['name']) + '\n揪團名稱:' + str(event_join['name']) + '\n揪團內容:' + str(event_join['describe']) + '\n揪團時間:' + str(event_join['time']) + '\n揪團人數:' + str(event_join['people']) + '\n揪團地點:' + str(event_join['location']) + '\n揪團Deadline:' + str(event_join['deadline'])
            )
            return message
        if stage == 8:
            event_delete = self.event_db.delete_user_participation(text[text.find('刪除')+2:], user_id)
            message = TextSendMessage(
                text='您已刪除 '+ event_delete['name'] 
            )
            return message
        if stage == 9:
            message = TextSendMessage(text='請輸入想加入的活動類別' )
            return message
        if stage == 10:
            event_new = {}
            event_new['user_id'] = user_id
            event_new['name'] = text[text.find('揪團名稱:')+5:text.find('\n', text.find('揪團名稱:'), text.find('揪團內容:'))]
            event_new['describe'] = text[text.find('揪團內容:')+5:text.find('\n', text.find('揪團內容:'), text.find('揪團時間:'))]
            event_new['time'] = text[text.find('揪團時間:')+5:text.find('\n', text.find('揪團時間:'), text.find('揪團人數:'))]
            event_new['people'] = '1/' + text[text.find('揪團人數:')+5:text.find('\n', text.find('揪團人數:'), text.find('揪團地點:'))]
            event_new['location'] = text[text.find('揪團地點:')+5:text.find('\n', text.find('揪團地點:'), text.find('揪團Deadline:'))]
            event_new['deadline'] = text[text.find('揪團Deadline:')+11: ]
            event_new['event_id'] = str(uuid.uuid4())
            event_new['image_url'] = 'https://example.com/item1.jpg'
            self.event_db.insert_data(event_new)
            message = TextSendMessage(text='您已成功發起活動')
            return message
        if stage == 11:
            action_col = []
            # read data from mock_data.py
            for course in mock_course_data.keys():
                action_col.append(
                    MessageTemplateAction(
                        label=str(course),
                        text='課程名稱:' + str(course)
                    )
                )
            
            message = TemplateSendMessage(
                alt_text='Buttons template',
                template=ButtonsTemplate(
                    # thumbnail_image_url='https://www.google.com/url?sa=i&url=https%3A%2F%2Fjome17.com%2F&psig=AOvVaw2yQ7E8QjPrYZ16KEJCGPld&ust=1697792244088000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCIiA2MH2gYIDFQAAAAAdAAAAABAI',
                    title='課程資訊',
                    text='Which course do you want to join?',
                    actions=action_col
                )
            )
            return message
        if stage == 12:
            print('text', text)
            print('text[text.find(\'課程名稱:\')+5:]', text[text.find('課程名稱:')+5:])
            message = TemplateSendMessage(
                alt_text = 'Buttons template',
                template = ButtonsTemplate(
                    # thumbnail_image_url='https://example.com/item1.jpg',
                    title='課程資訊-' + str(text[text.find('課程名稱:')+5:]),
                    text = '請選擇查看作業或是課堂討論',
                    actions=[
                        MessageTemplateAction(
                            label='私人助教',
                            text='查看作業:' + str(text[text.find('課程名稱:')+5:]) ## course name    
                        ),
                        MessageTemplateAction(
                            label='課程討論室',
                            text='課堂討論:'+ str(text[text.find('課程名稱:')+5:]) ## course name
                        ),
                    ]
                )
            )
            print('message', message)
            return message
        if stage == 13:
            course_name = text[text.find('查看作業:')+5:]
            hws = mock_course_data[course_name]['homeworks']
            all_carousel_column = []
            #user_events = event_db.print_events_for_user(event.source.user_id)
            for hw in range(len(hws)):
                # read txt file
                #hw_description = open(hws[hw]['description_path'], "r").read()
                hw_description = 'hw_description'
                hw_description_str = str(hw_description)
                if len(hw_description_str) > 57:
                    hw_description_str = hw_description_str[:57] + '...'
                carousel_column = CarouselColumn(
                    # thumbnail_image_url='https://example.com/item1.jpg',
                    title=str(hws[hw]['title']),
                    text=hw_description_str,
                    actions=[
                        MessageTemplateAction(
                            label='即時問答',
                            text='QA: ' + str(course_name) + ' -' +str(hws[hw]['title'])
                        )
                    ]
                )
                all_carousel_column.append(carousel_column)
            message = TemplateSendMessage(
                alt_text='Carousel template',
                template=CarouselTemplate(
                    columns=all_carousel_column
                )
            )
            return message
        if stage == 14:
            course_name, hw_name = text.split()[1], ' '.join(text.split()[2:])[1:]
            print(course_name, hw_name)
            hw_description_path = [hw['description_path'] for hw in mock_course_data[course_name]['homeworks'] if hw['title']==hw_name][0]
            with open(hw_description_path, 'r', encoding="utf-8") as f:
                hw_summary = f.readlines()
            
            self.event_db.set_course_question_prefix(user_id, hw_summary)

            message = TextSendMessage(
                text='作業總結: \n' + ' '.join(hw_summary) + '\n\n如果想要結束QA，請按下方"結束問答"，謝謝!',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="結束問答", text="結束問答")
                        )
                    ]
                )
            )
            return message
        if stage == 15:
            course_name = text[text.find('課堂討論:')+5:]
            message = TemplateSendMessage(
                alt_text = 'Buttons template',
                template=ButtonsTemplate(
                    # thumbnail_image_url='https://example.com/item1.jpg',
                    title='課堂討論-' + str(course_name),
                    text = '請選擇課堂紀錄或是課程聊天室',
                    actions=[
                        MessageTemplateAction(
                            label='過往紀錄',
                            text='課堂紀錄:' + str(course_name) ## course name    
                        ),
                        MessageTemplateAction(
                            label='線上討論',
                            text='課程聊天室:' + str(course_name) ## course name
                        )
                    ]
                )
            )
            return message
        if stage == 16:
            message = TextSendMessage(text=str('https://liff.line.me/2001167081-MwVpzVkx'))
            return message 
        if stage == 17:
            course_name = text[text.find('課堂紀錄:')+5:]
            chats = mock_course_data[course_name]['course_chats']
            actions = []
            for chat in range(len(chats)):
                actions.append(
                    MessageTemplateAction(
                        label=str(chats[chat]['time']),
                        text='正在產生彙整紀錄: ' + str(course_name) + ' ' + str(chats[chat]['time'])
                    )
                )
            message = TemplateSendMessage(
                alt_text='Buttons template',
                template=ButtonsTemplate(
                    # thumbnail_image_url='https://example.com/item1.jpg',
                    title='課堂紀錄-' + str(course_name),
                    text='請選擇想要的課堂紀錄日期',
                    actions=actions
                )
            )
            return message 
        if stage == 18:
            course_name, time = text.split()[1], text.split()[2]
            chats = mock_course_data[course_name]['course_chats']
            for chat in range(len(chats)):
                if chats[chat]['time'] == time:
                    message = TextSendMessage(text=str(chats[chat]['summary']))
            return message
        if stage == -1:
            self.event_db.set_course_question_prefix(user_id, "")
            return TextSendMessage(text='結束問答')
        if stage == 19:
            message = TemplateSendMessage(
                alt_text='Buttons template',
                template=ButtonsTemplate(
                    # thumbnail_image_url='https://www.google.com/url?sa=i&url=https%3A%2F%2Fjome17.com%2F&psig=AOvVaw2yQ7E8QjPrYZ16KEJCGPld&ust=1697792244088000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCIiA2MH2gYIDFQAAAAAdAAAAABAI',
                    title='失物招領',
                    text='請選擇想要的選項',
                    actions=[
                        MessageTemplateAction(
                            label='尋找失物',
                            text='我想要找東西'
                        ),
                        MessageTemplateAction(
                            label='張貼失物',
                            text='我想要張貼東西'
                        )
                    ]
                )
            )
            return message
        if stage == 20:
            message = TextSendMessage(text='請輸入以下資訊:\n1.物品名稱:[請填入]\n2.物品地點:[請填入]\n3.LINE ID:[請填入]\n4.物品描述:[請填入]')
            return message
        if stage == 21:
            item_new = {}
            item_new['user_id'] = user_id
            item_new['name'] = text[text.find('物品名稱:')+5:text.find('\n', text.find('物品名稱:'), text.find('物品地點:'))]
            item_new['location'] = text[text.find('物品地點:')+5:text.find('\n', text.find('物品地點:'), text.find('LINE ID:'))]
            item_new['line_id'] = text[text.find('LINE ID:')+8:text.find('\n', text.find('LINE ID:'), text.find('物品描述:'))]
            item_new['describe'] = text[text.find('物品描述:')+5:]
            item_new['item_id'] = str(uuid.uuid4())
            infos = '物品名稱:' + item_new['name'] + '\n物品地點:' + \
                    item_new['location'] + '\nLINE ID:' + item_new['line_id'] + \
                    "\n物品描述:" + item_new['describe'] + "\nUniqueID:" + item_new['item_id']
            message = TextSendMessage(text='您已張貼' + item_new['name'] + '\n物品名稱:' + item_new['name'] + 
                                      '\n物品地點:' + item_new['location'] + '\nLINE ID:' + item_new['line_id'] + "\n物品描述:" + 
                                      item_new['describe'])
            # TODO: insert pinecone db
            self.Pinecone_DB.add_text_to_index(infos)
            # new item db
            return message
        if stage == 22:
            message = TextSendMessage(text='請輸入物品名稱:')
            return message
        if stage == 23:
            # TODO: search pinecone db
            results = self.Pinecone_DB.search_document("學生證", topN=5)
            print('results', results)
            # item top is a list of dictionary
            item_top = []
            for result in results:
                result = result.split('\n')
                item_top.append({'name': result[0], 'location': result[1], 'line_id': result[2], 'describe': result[3], 'item_id': result[4]})
            all_carousel_column = []
            for i in range(len(item_top)):
                img_url = "https://example.com/item1.jpg"
                all_carousel_column.append(
                    CarouselColumn(
                        # thumbnail_image_url=img_url,
                        title=item_top[i]['name'],
                        text=str(item_top[i]['location']) + "\n" + str(item_top[i]['describe']) ,
                        actions=[
                            MessageTemplateAction(
                                label='這是我的東西',
                                text='這是我的東西:' + str(item_top[i]['item_id'])
                        )]
                    ))
            message = TemplateSendMessage(
                alt_text='Carousel template',
                template=CarouselTemplate(
                    columns=all_carousel_column
                )
            )
            #message = TextSendMessage(text='')
            return message
        if stage == 24:
            item_id = text[text.find('這是我的東西:')+7:]
            item_delete = {}
            # TODO: Fix the exact search
            # TODO: Delete pinecone db
            results = self.Pinecone_DB.search_document("學生證", topN=1)[0]
            results = results.split('\n')
            item_delete = {'name': results[0], 'location': results[1], 'line_id': results[2], 'describe': results[3], 'item_id': results[4]}
            message = TextSendMessage(text='您可以加line ID做為聯絡方式 '+ str(item_delete['line_id']))

            return message
        
        if stage == 25:
            prev_response = str(self.event_db.get_course_question_prefix(user_id))
            print(prev_response)

            qa_response = self.qa_agent(
                'Prev Answer: '+
                prev_response+
                '\nQuery: '+
                text+
                'please make anwser short.'
            )['response']

            self.event_db.set_course_question_prefix(user_id, qa_response)

            return TextSendMessage(
                text=qa_response,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label="結束問答", text="結束問答")
                        )
                    ]
                ),
            )
                        
# 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    db_name = 'event2'
    table_name = 'events'
    user_table_name = 'users'
    #event_db.clear_table()
    text = event.message.text
    actionchooser = ActionChooser(db_name='eventv3', table_name='events', user_table_name='users')
    message = actionchooser.run_chooser(event)
    print(message)
    line_bot_api.reply_message(event.reply_token, message)





if __name__ == "__main__":
    COURSE_SUMMARY = '''* Course summary:
[empty]
* Questions asked:
[empty]
'''
    mock_course_data = {
        "強化學習":{
            # HW
            "homeworks":[
                {
                    "title": "TD Learning",
                    "description_path": "./mock_data/rl_hw_0.txt",
                },
                {
                    "title": "Deep Q-Network",
                    "description_path": "./mock_data/rl_hw_1.txt",
                },
            ],
            "course_chats":[
                {
                    "time": "2023-10-21",
                    "summary": COURSE_SUMMARY,
                },
            ],
            "live_chatroom": "https://liff.line.me/2001167081-MwVpzVkx"
        },
        "作業系統":{
            # HW
            "homeworks":[
                {
                    "title": "Compiling Linux Kernel",
                    "description_path": "./mock_data/os_hw_0.txt"
                },
            ],
            # Discussion
            "course_chats":[
                {
                    "time": "2023-10-21",
                    "summary": COURSE_SUMMARY
                },
            ],
            "live_chatroom": "https://liff.line.me/2001167081-MwVpzVkx",
        },
        "演算法":{
            # HW
            "homeworks":[
                {
                    "title": "Proof Master Theorem",
                    "description_path": "./mock_data/algo_hw_0.txt"
                },
            ],
            # Discussion
            "course_chats":[
                {
                    "time": "2023-10-21",
                    "summary": COURSE_SUMMARY
                },
            ],
            "live_chatroom": "https://liff.line.me/2001167081-MwVpzVkx",
        },
    }
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    stage_22_flag = True
    