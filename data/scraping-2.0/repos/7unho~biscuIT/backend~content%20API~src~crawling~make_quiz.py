# import config
from collections import deque
import os
import openai
import config

OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY


def make_quiz_api(text):
    model = "gpt-3.5-turbo"
    query = (
        "korean text : " + text
        + " format : '질문{i}번:{질문 내용} \n선택지1:{선택지1 내용} \n 선택지2:{선택지2 내용} \n 선택지3:{선택지3 내용} \n답:{선택지 내용} '"
        + "question: please make five questions with three multiple choices and an one answer in korean about korean text with fixed format"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    response = openai.ChatCompletion.create(model=model, messages=messages)

    res = response["choices"][0]["message"]["content"]
    print(res)
    return res

def insert_quiz(content_id, question, choices, answer):
    conn = config.connect()
    curs = conn.cursor()
    
    sql = "insert into quiz (content_id, question, multiple_choice, answer) values (%s, %s, %s, %s)"
    val = (content_id, question, choices, answer)
    try :
        curs.execute(sql, val)
        conn.commit()
    except : 
        config.close(conn)
        return
    config.close(conn)

def find_content_Id(title):
    conn = config.connect()
    curs = conn.cursor()
    global cId

    sql = "insert into quiz (content_id, question, multiple_choice, answer) values (%s, %s, %s, %s)"
    val = (content_id, question, choices, answer)
    try :
        curs.execute(sql, val)
        conn.commit()
    except : 
        config.close(conn)
        return
    config.close(conn)
    return cId

def find_content_Id(title):
    conn = config.connect()
    curs = conn.cursor()
    global cId

    sql = "select id from content where title = %s"
    curs.execute(sql, (title))
    result = curs.fetchall();

    for x in result:
        cId = x[0]
    config.close(conn)
    return cId

base_path = "data/content"
os.getcwd()
content_dir_list = os.scandir(base_path)
documents = []
quiz_dict = dict()
for content_dir in content_dir_list:
    if not content_dir.is_dir():
        continue
    content_list = os.scandir(content_dir)
    for content in content_list:
        if not content.is_file():
            continue
        basename = os.path.basename(content)
        if(basename == "index.txt") :
                continue;
        filename = os.path.splitext(basename)[0].split('_', 1)[1]
        print("filename : " + filename)
        try:
            content_id = find_content_Id(filename)
        except:
            continue
        if content_id < 1450 :
            continue
        file = open(content.path, "r", encoding="utf-8")
        text = file.read()
        try : 
            res = make_quiz_api(text)
        except:
            continue
        quizzes = deque(res.split("\n"))
        quizQuestion = ""
        choices = ""
        answer = ""
        while quizzes:
            question = quizzes.popleft()
            question = question.split(' ', 1)[1]
            quizQuestion = question
            quiz_dict[question] = []
            line = question
            idx = 0
            quizQuestion = ""
            choices = ""
            answer = ""
            while line and quizzes:
                line = quizzes.popleft()
                if(len(line.split(' ', 1)) > 1) :
                    tmpLine = line.split(' ', 1)[1]
                    quiz_dict[question].append(tmpLine)
                    if idx == 3 :
                        answer = tmpLine
                    else :
                        choices += tmpLine
                        choices += "//"
                idx+=1
            # for key, value in quiz_dict.items():
            #             print(key, value)
            # print("질문 : " + question + " 보기 : " + choices + " " + "답: " + answer)
            insert_quiz(content_id, question, choices, answer)