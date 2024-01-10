import openai
import json
from RAG import add_database, search_docs
from summary import summary_processor
import os
import shutil
import ast
from prompt_function import rich_character, character, rich_place, summary
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#### global variable 
m_list = [[],[],[],[]] 
demo_story_count=1
demo_mode = True
#### global DB
db_0 = Chroma()  ## story DB
db_1 = Chroma()  ## suspect 1 DB
db_2 = Chroma()  ## suspect 2 DB
db_3 = Chroma()  ## suspect 3 DB
db_4 = Chroma()  ## scene DB
db_len = [1,1,1,1,1]
embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def get_current_db():
    global db_0, db_1, db_2, db_3, db_4
    global db_len
    #### check story DB
    if(os.path.exists('./db/story')):
        db_0 = Chroma(persist_directory='./db/story', embedding_function=embedding_function)
    else:
        print("story DB has some problem")
        print("Recreate story DB")
        db_0, db_len[0] = add_database("./story_background/story.txt", './db/story', 1)
    suspect_db = 0
    #### check suspect DB
    if(os.path.exists('./db/suspect_1')):
        db_1 = Chroma(persist_directory='./db/suspect_1', embedding_function=embedding_function)
        suspect_db += 1
    if(os.path.exists('./db/suspect_2')):
        db_2 = Chroma(persist_directory='./db/suspect_2', embedding_function=embedding_function)
        suspect_db += 1
    if(os.path.exists('./db/suspect_3')):
        db_3 = Chroma(persist_directory='./db/suspect_3', embedding_function=embedding_function)
        suspect_db += 1
    if(suspect_db != 3):
        print("suspect DB has some problem")
        print("Recreate suspect DB")
        db_len = rich_character_info()
    #### check scene DB
    if(os.path.exists('./db/scene')):
        db_4 = Chroma(persist_directory='./db/scene', embedding_function=embedding_function)
    else:
        print("scene DB has some problem")
        print("Recreate scene DB")
        rich_scene_info()
    db_len = [len(db_0.get()['ids']),len(db_1.get()['ids']),len(db_2.get()['ids']),len(db_3.get()['ids']),len(db_4.get()['ids'])]
    print("Get current db")
    print("DB len:")
    print(db_len)
    print("Please make sure every number in DB len is bigger than 1 (not including 1).")

def reset_db():
    global db_0, db_1, db_2, db_3, db_4
    global db_len
    db_0 = Chroma()  ## story DB
    db_1 = Chroma()  ## suspect 1 DB
    db_2 = Chroma()  ## suspect 2 DB
    db_3 = Chroma()  ## suspect 3 DB
    db_4 = Chroma()  ## scene DB
    db_len = [1,1,1,1,1]
    ### delete all file in db
    dirPath = "./db"
    try:
        shutil.rmtree(dirPath)
    except OSError as e:
        print(f"Error:{ e.strerror}")
    print("Reset db")
    print("DB len:")
    print(db_len)


#### openai processor
def GPT_processor(length,system_message, user_message, function_description, temperature):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=length,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
            ],
        functions=function_description,
        function_call = {"name":function_description[0]['name']}
    )
    return response

###### game creator
def story_creater():
    # conversation_count = 0
    m_list = [[],[],[],[]] 
    #### load start prompt
    f = open('./story_background/start_prompt.txt',encoding="utf-8")
    start_prompt = f.read()
    with open('./story_background/story_background_description.json',encoding="utf-8") as file:
        story_background_description = [json.load(file)]
    print("start process AI api")
       
    story_response = GPT_processor(2000,start_prompt, "我想要玩一個偵探尋找殺人犯的遊戲，你可以幫我設計一個遊戲的故事內容嗎？", story_background_description, 0.5)
    # store the story dictionary result to story.json
    f2 = open("./story_background/story.txt", "w",encoding="utf-8")
    f2.write(story_response.choices[0].message.function_call.arguments)
    f2.close()
    story_object  = json.dumps(story_response.choices[0].message.function_call.arguments)
    # print(story_object.encode('ascii').decode('unicode-escape'))
    with open("./story_background/story.json", "w") as outfile:
        outfile.write(story_object)
    
    #### add story to DB
    print("Add story to DB")
    global db_0, db_len
    db_0, db_len[0] = add_database("./story_background/story.txt", './db/story', 1)
    print("DB status")
    print(db_len)

    #### rich character info
    print("rich character info")
    db_len = rich_character_info()
    print("DB status")
    print(db_len)

    #### rich scene info
    print("rich scene info")
    rich_scene_info()
    print("DB status")
    print(db_len)

    
    return story_response.choices[0].message.function_call.arguments

def rich_character_info():
    f = open("./story_background/story.json")
    story_json = json.load(f)
    suspects_list = json.loads(story_json).get("嫌疑人")
    #### load start prompt
    with open('./suspect_file/rich_suspect_prompt.txt', encoding="utf-8") as f:
        rich_prompt = f.read()
    with open('./story_background/story.txt', encoding="utf-8") as f:
        story = f.read()

    for i in range(3):
        name = suspects_list[i]["姓名"]
        rich_character_response = rich_character(rich_prompt, story, name, 1300, 0.5)
        with open(f'./suspect_file/suspect_{i}/rich_suspect_info.txt', 'w', encoding="utf-8") as f:
            f.write(rich_character_response)
        if(i == 0):
            global db_1
            db_1, db_len[1] = add_database(f'./suspect_file/suspect_{i}/rich_suspect_info.txt', './db/suspect_1', 1)
        elif(i == 1):
            global db_2
            db_2, db_len[2] = add_database(f'./suspect_file/suspect_{i}/rich_suspect_info.txt', './db/suspect_2', 1)
        else:
            global db_3
            db_3, db_len[3] = add_database(f'./suspect_file/suspect_{i}/rich_suspect_info.txt', './db/suspect_3', 1)
    return db_len
    

###### suspect creater
def suspect_creater(id,target,action):
    global db_1, db_2, db_3
    global db_len
    #### load suspect prompt
    f = open('./suspect_file/suspect_prompt.txt',encoding="utf-8")
    suspect_prompt = f.read()

    keyword = action + "," + target
    
    #### load story txt
    with open('./story_background/story.txt',encoding="utf-8") as file:
        story = file.read()
    
    suspect_db_1 = Chroma(persist_directory=f'./db/suspect_1', embedding_function=embedding_function)
    suspect_db_2 = Chroma(persist_directory=f'./db/suspect_2', embedding_function=embedding_function)
    suspect_db_3 = Chroma(persist_directory=f'./db/suspect_3', embedding_function=embedding_function)
    if(id == 0):
        suspect_summary = search_docs(keyword, suspect_db_1, 5)
    elif(id == 1):
        suspect_summary = search_docs(keyword, suspect_db_2, 5)
    else:
        suspect_summary = search_docs(keyword, suspect_db_3, 5)
    user_prompt = action

    system_prompt = suspect_prompt + "你需要假扮嫌疑人是" + target + "提供的資訊如下:" + suspect_summary + "\n" + story
    with open('./suspect_file/suspect_description.json',encoding="utf-8") as file:
        suspect_description = [json.load(file)]
    suspect_response = GPT_processor(200,system_message=system_prompt, user_message=user_prompt, function_description=suspect_description, temperature=0.5)
    # suspect_object  = json.dumps(suspect_response.choices[0].message.function_call.arguments)
    
    #### store to list
    conversation_information = {
        "m_id": len(m_list[id]),
        "sender": 1,
        "message":action
    }
    m_list[id].append(conversation_information)

    conversation_information = {
        "m_id": len(m_list[id]),
        "sender": 0,
        "message":json.loads(suspect_response.choices[0].message.function_call.arguments).get("回覆")
    }
    m_list[id].append(conversation_information)
    ### write to conversation file
    f = open(f"./suspect_file/suspect_{id}/conversation.txt", "a",encoding="utf-8")
    f.write(f"玩家:{action}\n")
    f.write( f"{target}:{ json.loads(suspect_response.choices[0].message.function_call.arguments).get('回覆') }\n" )
    f.close()
    ### save to DB
    with open(f"./suspect_file/suspect_{id}/tmp_conversation.txt", "w" ,encoding="utf-8") as file:
        file.write(f"玩家:{action}\n")
        file.write( f"{target}:{ json.loads(suspect_response.choices[0].message.function_call.arguments).get('回覆') }\n" )
    if(id == 0):
        db_1, d1_len = add_database(f"./suspect_file/suspect_{id}/tmp_conversation.txt", './db/suspect_1', db_len[1])
        db_len[1] += d1_len
    elif(id == 1):
        db_2, d2_len = add_database(f"./suspect_file/suspect_{id}/tmp_conversation.txt", './db/suspect_2', db_len[2])
        db_len[2] += d2_len
    else:
        db_3, d3_len = add_database(f"./suspect_file/suspect_{id}/tmp_conversation.txt", './db/suspect_3', db_len[3])
        db_len[3] += d3_len
    

    open(f"./suspect_file/suspect_{id}/tmp_conversation.txt", 'w').close()
    
    # summary_processor()
    print(json.loads(suspect_response.choices[0].message.function_call.arguments).get("回覆"))

    return json.loads(suspect_response.choices[0].message.function_call.arguments).get("回覆")

    

def rich_scene_info():
    with open('./scene_file/rich_scene_prompt.txt', encoding="utf-8") as f:
        rich_prompt = f.read()
    with open('./story_background/story.txt', encoding="utf-8") as f:
        story = f.read()
    rich_scene = rich_place(rich_prompt, story, 1300, 0.8)

    with open('./scene_file/rich_scene_info.txt', 'w', encoding="utf-8") as f:
        f.write(rich_scene)
    global db_4, db_len
    db_4, db_len[4] = add_database(f"./scene_file/rich_scene_info.txt", './db/scene', 1)
    
###### scene_creater
def scene_creater(action):
    global db_4, db_len
    #### load start prompt
    f = open('./scene_file/scene_prompt.txt',encoding="utf-8")
    scene_prompt = f.read()

    keyword = "請根據案發現場的狀況，回答：" + action
    #### load story txt
    with open('./story_background/story.txt',encoding="utf-8") as file:
        story = file.read()
    
    scene_db = Chroma(persist_directory='./db/scene', embedding_function=embedding_function)
    scene_info = search_docs(keyword, scene_db, 8)
    
    system_prompt = scene_prompt + story  + "\n" + scene_info
    user_prompt = action
    with open('./scene_file/scene_description.json',encoding="utf-8") as file:
        scene_description = [json.load(file)]
    scene_response = GPT_processor(400,system_message=system_prompt, user_message=user_prompt, function_description=scene_description, temperature=0.9)
    
    #### store to list
    conversation_information = {
        "m_id": len(m_list[3]),
        "sender": 1,
        "message":action
    }
    m_list[3].append(conversation_information)

    conversation_information = {
        "m_id": len(m_list[3]),
        "sender": 0,
        "message":json.loads(scene_response.choices[0].message.function_call.arguments).get("回覆")
    }
    m_list[3].append(conversation_information)


    # scene_object  = json.dumps(scene_response.choices[0].message.function_call.arguments)
    f = open("./scene_file/conversation.txt", "a",encoding="utf-8")
    f.write(f"玩家:{action}\n")
    f.write( f"場景:{ json.loads(scene_response.choices[0].message.function_call.arguments).get('回覆') }\n" )
    print(json.loads(scene_response.choices[0].message.function_call.arguments).get("回覆"))
    f.close()

    #### save to DB
    with open(f"./scene_file/tmp_conversation.txt", "w" ,encoding="utf-8") as file:
        file.write(f"玩家:{action}\n")
        file.write( f"場景:{ json.loads(scene_response.choices[0].message.function_call.arguments).get('回覆') }\n" )
    print("before add scene to DB")
    print(db_len)
    db_4, d4_len = add_database(f"./scene_file/tmp_conversation.txt", './db/scene', db_len[4])
    db_len[4] += d4_len
    print("after add scene to DB")
    print(db_len)
    open(f"./scene_file/tmp_conversation.txt", 'w').close()
    return json.loads(scene_response.choices[0].message.function_call.arguments).get("回覆")
    # return scene_response.choices[0].message.content

###### final_answer_creater
def final_answer_creater(id, motivation, action):
    #### load start prompt
    f = open('./final_answer_file/final_answer_prompt.txt',encoding="utf-8")
    fa_prompt = f.read()

    #### load story txt
    with open('./story_background/story.txt',encoding="utf-8") as file:
        story = file.read()
    f = open("./story_background/story.json")
    story_json = json.load(f)
    suspects_list = json.loads(story_json).get("嫌疑人")
    system_prompt = fa_prompt + '\n' + story
    user_prompt = "我猜測兇手是" +  suspects_list[id]["姓名"] + "，他的動機為" + motivation + "，他的犯案手法為" + action
    with open('./final_answer_file/final_answer_description.json',encoding="utf-8") as file:
        final_answer_description = [json.load(file)]
    final_answer_response = GPT_processor(800,system_message=system_prompt,user_message=user_prompt, function_description=final_answer_description, temperature=0.8)
    print(json.loads(final_answer_response.choices[0].message.function_call.arguments).get("真相"))
    return json.loads(final_answer_response.choices[0].message.function_call.arguments).get("真相")

#### hint creater  
def hint_creater():
    #### load start prompt
    f = open('./story_background/hint_prompt.txt',encoding="utf-8")
    hint_prompt = f.read()


    #### load story txt
    with open('./story_background/story.txt',encoding="utf-8") as file:
        story = file.read()
    system_prompt =   hint_prompt + '\n' + story
    user_prompt = "請給我一個提示。"
    with open('./story_background/hint_description.json',encoding="utf-8") as file:
        hint_description = [json.load(file)]
    hint_response = GPT_processor(200,system_prompt, user_prompt, hint_description, 0.6)
    
    f = open(f"./story_background/hints_history.txt", "a",encoding="utf-8")
    f.write(json.loads(hint_response.choices[0].message.function_call.arguments).get("回覆"))
    f.close()
    print( json.loads(hint_response.choices[0].message.function_call.arguments).get("回覆") )

    return json.loads(hint_response.choices[0].message.function_call.arguments).get("回覆")



























# def message_creater(id):
#     file = []
#     if id < 3:
#         file = open(f"./suspect_file/suspect_{id}/conversation.txt",encoding="utf-8")
#     else:
#         file = open(f"./scene_file/conversation.txt",encoding="utf-8")
#     conversation_list = file.readlines() #list
#     for lines in conversation_list:

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#   .function_call.arguments).get("place")
# print(story_response.choices[0].message.content)
# story_background = story_response.choices[0].message.content
##### create information dictionary
# dictionary_response = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   max_tokens=2000,
#   temperature=0.5,
#   messages=[
#         {"role": "user", "content": story_response.choices[0].message.content},
#         {"role": "user", "content": "請幫我將上述偵探小說內容根據每個角色的人格特質和線索做出一個dictionary"},
#         # {"role": "assistant", "content": "原來你是楊鈞安呀"},
#         # {"role": "user", "content": "請問我叫什麼名字？"}
#     ]
# )
# # store the dictionary to data_dictionary
# print(dictionary_response.choices[0].message.content)
# with open('data_dictionary.txt', 'w') as convert_file: 
#      convert_file.write(json.dumps(dictionary_response.choices[0].message.content))

# ##### 接收前端回傳資料 
# # from flask import Flask, request

# # app = Flask(__name__)

# # @app.route('/receive_message', methods=['POST'])
# # def receive_message():
# #     reply = request.get_json()
# #     # 处理接收到的消息
# #     return 'Message received successfully'

# ##### game loop
# while True:
#     action = input("action:")
#     f = open('data_dictionary.txt',encoding="utf-8")
#     now_dictionary = f.read()
#     game_response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     max_tokens=200,
#     temperature=0.5,
#     messages=[
#             {"role": "user", "content": story_background},
#             {"role": "user", "content": "請根據以下關係資料和上述故事內容，進行本偵探遊戲，請勿直接公布兇手身分。"+now_dictionary+" 若是資料中無法確定的資訊，請你幫我生成有邏輯和不矛盾的內容。" },
#             {"role": "user", "content": action},
#             # {"role": "assistant", "content": "原來你是楊鈞安呀"},
#             # {"role": "user", "content": "請問我叫什麼名字？"}
#         ]
#     )
#     # store the dictionary to data_dictionary
#     # print(dictionary_response.choices[0].message.content)
#     # with open('data_dictionary.txt', 'w') as convert_file: 
#     #     convert_file.write(json.dumps(dictionary_response.choices[0].message.content))
#     print(f"response: {game_response.choices[0].message.content}")
#     ##### dictionary reset
#     dictionary_response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     max_tokens=200,
#     temperature=0.5,
#     messages=[
#             {"role": "user", "content": story_background},
#             {"role": "user", "content": game_response.choices[0].message.content + " 請幫我將上述內容根據每個角色的人格特質和線索做出一個dictionary，並和以下dictionary合併成一個新的關係dictionary"+ now_dictionary },
#             # {"role": "user", "content": action},
#             # {"role": "assistant", "content": "原來你是楊鈞安呀"},
#             # {"role": "user", "content": "請問我叫什麼名字？"}
#         ]
#     )
#     # store the dictionary to data_dictionary
#     print(dictionary_response.choices[0].message.content)
#     with open('data_dictionary.txt', 'w') as convert_file: 
#         convert_file.write(json.dumps(dictionary_response.choices[0].message.content))