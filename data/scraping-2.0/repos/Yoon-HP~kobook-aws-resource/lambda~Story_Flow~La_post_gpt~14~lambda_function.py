import json
import os
import openai
import boto3
import traceback
import random
import time
import re
import urllib.request
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from post_slack import post_slack

load_dotenv("key.env")

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPEN_API_KEY")


def make_stroy(
    messages, temperature=1, top_p=1, n=1, presence_penalty=0, frequency_penalty=0
):
    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    output = chat_response["choices"][0]["message"]["content"]
    return output


def make_stroy_gpt_4(
    messages, temperature=1, top_p=1, n=1, presence_penalty=0, frequency_penalty=0
):
    chat_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    output = chat_response["choices"][0]["message"]["content"]
    return output


def post_slack_book(argStr):
    message = argStr
    send_data = {
        "text": message,
    }
    send_text = json.dumps(send_data)
    request = urllib.request.Request(
        "slack webhook url",
        data=send_text.encode("utf-8"),
    )
    # slack webhook url

    with urllib.request.urlopen(request) as response:
        slack_message = response.read()


# sqs에서 전달받은 user_id, timestamp 토대로 글 생성에 필요한 데이터를 쿼리 해오기
def lambda_handler(event, context):
    # test code 확인
    print(event)
    # version check
    function_arn = context.invoked_function_arn
    env = function_arn.split(":")[-1]
    # just concurreny
    env = "prod"
    dynamodb_client = boto3.client("dynamodb")
    # event parsing
    try:
        # print(event)
        # print(event)
        message_body = json.loads(event["Records"][0]["body"])
        # bucket_name=message_body['Records'][0]['s3']['bucket']['name']
        # object_key=message_body['Records'][0]['s3']['object']['key']
        user_id = message_body["user_id"]
        time_stamp = message_body["time_stamp"]
        try:
            version = message_body["version"]
        except:
            version = "1"
        print(version)

        start_time = time.time()
        # slack 노티
        try:
            post_slack_book(f"book make request occur! {user_id}, {time_stamp}")
        except:
            print("noti error!")
        end_time = time.time()
        print(f"post_slack: {end_time - start_time:.5f}")

    except:
        print("why??")

    if version == "1":
        try:
            # print(user_id,time_stamp)
            query = f"SELECT gender,age,name,major,middle,sub FROM Dy_story_event_{env} where user_id='{user_id}' and time_stamp='{time_stamp}'"
            # print(query)
            result = dynamodb_client.execute_statement(Statement=query)
            # print(result)
            # print("check!!")

            # 다시 N으로 바꿔줘야함
            age = str(result["Items"][0]["age"]["S"])
            gender = result["Items"][0]["gender"]["S"]
            name = result["Items"][0]["name"]["S"]
            major = result["Items"][0]["major"]["S"]
            middle = result["Items"][0]["middle"]["S"]
            sub = result["Items"][0]["sub"]["S"]

        except:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: event_parsing_fail!"
            post_slack(notification)
            print("event_parsing_fail!!!!!")
            return {"statusCode": 200, "body": "..."}
        # 글 생성

        message = [
            {
                "role": "user",
                "content": "너는 아이들을 위한 동화 생성하고 이야기해 주는 작가 선생님으로, 아이들에게 이야기해주는 톤을 유지하며, 아래의 캐릭터설정을 기반으로 동화분류와 제약사항을 참고하여 그 구성과 아래 출력형식(JSON)에 맞춰 동화를 작성해라.\n\n동화분류:\n대분류: %(major)s;\n중분류:%(middle)s;\n소분류:%(sub)s\n\n제약사항:\n최대페이지수: 8;\n페이지수: 8;\n페이지의양: 최소 30 tokens;\n제목: 캐릭터의 이름이 들어가는 동화제목\n대상: 3~6세\n말투: 했어요;\n\n캐릭터설정:\n이름:%(name)s;\n나이:%(age)s살,\n성별:%(gender)s\n\n출력형식:{제목:동화의 제목, 1:페이지 내용, 2:페이지 내용, 3:페이지 내용, 4:페이지 내용, 5:페이지 내용, 6:페이지 내용, 7:페이지 내용, 8:페이지 내용}"
                % {
                    "major": major,
                    "middle": middle,
                    "sub": sub,
                    "name": name,
                    "age": age,
                    "gender": gender,
                },
            }
        ]

        # story=make_stroy(message,temperature=0.75)

        # print(message)

        # json 형식 검사
        temperature = 1
        flag = False
        while temperature > 0.7:
            try:
                story = make_stroy_gpt_4(message, temperature=temperature)
                print(story)
                story_json = json.loads(story)
                if len(list(story_json.keys())) == 9:
                    title = story_json[list(story_json.keys())[0]]
                    # print(title)
                    del story_json[list(story_json.keys())[0]]

                    f = True
                    # gpt에서 글이 생성되지 않는 경우 방지...
                    for key in list(story_json.keys()):
                        if len(story_json[key]) == 0:
                            f = False
                    if f:
                        flag = True
                        break
                else:
                    temperature -= 0.1
            except:
                try:
                    check = story.split("{")
                    if len(check[0]) > 1:
                        if len(check) == 2:
                            temp_story = "{" + check[1]
                            print(temp_story)
                            story_json = json.loads(temp_story)
                            try:
                                if len(list(story_json.keys())) == 9:
                                    title = story_json[list(story_json.keys())[0]]
                                    print(title)
                                    del story_json[list(story_json.keys())[0]]

                                    f = True
                                    # gpt에서 글이 생성되지 않는 경우 방지...
                                    for key in list(story_json.keys()):
                                        if len(story_json[key]) == 0:
                                            f = False
                                    if f:
                                        flag = True
                                        break
                            except:
                                pass
                except:
                    print("haha")

                temperature -= 0.1
                err_msg = traceback.format_exc()
                print("1", err_msg)

        if not flag:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: gpt_fail!!"
            post_slack(notification)
            print("gpt fail!!!")
            return {"statusCode": 200, "body": "gg"}
        try:
            story_image = {}

            story_josn_key = list(story_json.keys())

            for key_index in range(len(story_josn_key)):
                print(story_json[story_josn_key[key_index]])

                # {story_json[story_josn_key[key_index]]}

                message_image = [
                    {
                        "role": "system",
                        "content": f"너는 주어진 동화내용에 대한 삽화를 생성하는 삽화가이다. 주어진 동화내용을 이해한 후 이에 어울리는 장면을 영어명사구로 아래 제약사항에 맞춰 출력해라.\n\n제약사항:\n영어로 작성한다.\n반드시 캐릭터 이름을 성별에 따른 인칭대명사로 치환한다.\n캐릭터의 이름을 사용하지 않고 아래 성별을 참고해 인칭대명사로 표현한다.\n등장인물과 주변 객체 간의 행위에 초점을 둔다.\n고유명사를 제거한다.\n캐릭터의 행동을 묘사한다.\n절대 고유명사를 사용하지 않는다.\n반드시 캐릭터는 인간이다.\n절대 출력에 고유명사를 사용하지 않는다.\n출력에 들어가는 캐릭터의 이름을 인칭대명사로 바꾼다.\n\n캐릭터:\n이름:{name};\n나이:{age}살;\n성별:{gender};",
                    },
                    {
                        "role": "user",
                        "content": f"동화내용:\n어느 날, 꼬미가 밤하늘에 뜬 거대한 보름달을 보고 있었어요. '우와, 보름달이 정말 크고 둥글다!' 꼬미는 감탄했어요.\n출력:\na young girl, happy, looking at moon, astonishment\n동화내용:\n어머낫!!, 날 볼 수 있나봐!' 토끼가 대답했어요. '나는 달토끼라고 해. 반가워 꼬미야!\n출력:\na white cute rabbit, on the moon surface, happy\n동화내용:\n그러다가 꼬미는 흐릿한 지구를 보았어요. '우리 집은 저기야!' 꼬미는 달토끼에게 자신의 이야기를 들려주었어요.\n출력:\nA young girl and a white cute rabbit pointing at the Earth from the moon surface\n동화내용:\n'북'이 병원에 도착하니, 움츠려있는 작은 새가 보였어요. 새의 날개가 부러진 것 같았어요. 북은 첫 환자를 치료해보려고 해봤어요.\n출력:\nA young boy is sitting a little bird in her arms.\n동화내용:\n'북'이 다시 새에게 돌아가서 섬세하게 날개를 진료했어요. 무언가 부서진 것을 찾아 고쳐주었답니다.\n출력:\nA young boy touches the small bird with her hands.\n동화내용:\n그래서 아이들, 우리도 환경구 군처럼 우리 환경을 사랑하고, 쓰레기는 쓰레기통에 버리는 걸 잊지 말아요. 안녕!\n출력:\nA young boy is smiling at the garbage dump.\n동화내용:\n{story_json[story_josn_key[key_index]]};\n출력:",
                    },
                ]

                # 무한루프에 빠지는 경우 존재
                cnt = 0
                while cnt < 3:
                    try:
                        image_prompt = make_stroy(message_image, temperature=1)

                        # 한글이 존재하면 최대 세번까지 다시 생성하도록
                        temp = image_prompt
                        word_check = re.sub(r"[^ㄱ-ㅣ가-힣\s]", "", temp)
                        word_check = word_check.replace(" ", "")

                        # 한국어가 생성된 것
                        if len(word_check) != 0:
                            cnt += 1
                            time.sleep(1)
                            print(image_prompt)
                            continue
                        break
                    except:
                        err_msg = traceback.format_exc()
                        print("2", err_msg)

                # 문제가 발생한 것
                if cnt == 3:
                    notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: 이미지 프롬프트 생성 이슈 (한국어 존재)!!"
                    post_slack(notification)

                story_image[story_josn_key[key_index]] = image_prompt
                print(image_prompt)

        except:
            err_msg = traceback.format_exc()
            print("3", err_msg)
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: check plz!!"
            post_slack(notification)
            print("something went wrong!!")
            return {"statusCode": 500, "body": "chatgpt error!!"}

        dynamodb = boto3.resource("dynamodb", region_name="ap-northeast-2")
        table_gpt_story = dynamodb.Table(f"Dy_gpt_story_{env}")
        table_gpt_prompt = dynamodb.Table(f"Dy_gpt_prompt_{env}")
        table_story_summary = dynamodb.Table(f"Dy_story_summary_prod")
        # dynamodb put!!
        try:
            # 스토리
            story_json["user_id"] = user_id
            story_json["time_stamp"] = time_stamp
            temp = table_gpt_story.put_item(Item=story_json)

            # 스토리 상황묘사
            story_image["user_id"] = user_id
            story_image["time_stamp"] = time_stamp
            temp = table_gpt_prompt.put_item(Item=story_image)

            # 책 제목 설정
            query = f"UPDATE Dy_story_event_{env} SET title = '{title}' WHERE user_id='{user_id}' and time_stamp='{time_stamp}';"
            result = dynamodb_client.execute_statement(Statement=query)

        except:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: dynamodb error!"
            post_slack(notification)
            print("put dynamodb error!!!!")

        # 줄거리 생성
        try:
            story = ""
            for key_index in range(len(story_josn_key)):
                story += " " + story_json[story_josn_key[key_index]]
            print(story)
            message = [
                {
                    "role": "user",
                    "content": "다음 동화를 이해하고 해당 동화에 대한 줄거리를 두 문장으로 작성해줘.\n\n동화:\n%(story)s;"
                    % {
                        "story": story,
                    },
                }
            ]
            while True:
                try:
                    story_summary = make_stroy(message, temperature=1)
                    break
                except:
                    err_msg = traceback.format_exc()
                    print("2", err_msg)

            print(story_summary)
            story_summary_json = {}
            story_summary_json["user_id"] = user_id
            story_summary_json["time_stamp"] = time_stamp
            story_summary_json["summary"] = story_summary
            temp = table_story_summary.put_item(Item=story_summary_json)

        except:
            print("error occur!!")

        # 퀴즈 생성
        table_story_quiz = dynamodb.Table(f"Dy_story_quiz_prod")

        message = [
            {
                "role": "user",
                "content": "너는 주어진 동화에 대한 퀴즈를 만들어주는 작가 선생님으로, 아이들에게 이야기해주는 톤을 유지하며, 아래의 동화제목, 동화내용을 기반으로 제약사항을 참고하여 아래 출력형식(JSON)에 맞춰 퀴즈를 생성해라.\n\n동화제목: %(title)s\n동화내용: %(story)s\n\n제약사항:\n최대 question 개수: 4개;\nquestion 개수: 4개;\nchoices수: 4개;\n출력형식:{quiz_title:"
                ",quiz_questions:{[question:"
                ", choices:"
                ",correct_answer:"
                "},{question:"
                ", choices:"
                ",correct_answer:"
                "},{question:"
                ", choices:"
                ",correct_answer:"
                "},{question:"
                ", choices:"
                ",correct_answer:}]}"
                % {
                    "title": title,
                    "story": story,
                },
            }
        ]
        quiz_temp = {}
        quiz_temp["user_id"] = user_id
        quiz_temp["time_stamp"] = time_stamp

        while True:
            try:
                story_quiz = make_stroy(message, temperature=1)
                story_json = json.loads(story_quiz)
                print(story_json["quiz_title"])
                print(story_json["quiz_questions"])
                number = 1
                for question_temp in story_json["quiz_questions"]:
                    question = question_temp["question"]
                    choices = question_temp["choices"]
                    correct_answer = question_temp["correct_answer"]

                    # 만약 정답이 int로 나왔다면 -> 정답으로 매칭
                    # choices는 배열
                    if type(correct_answer) == int:
                        notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: quiz check plz integer type issue!!"
                        post_slack(notification)
                        temp_answer = choices[correct_answer - 1]
                        correct_answer = temp_answer

                    print(choices)
                    quiz_temp[f"{number}_question"] = question
                    quiz_temp[f"{number}_choices"] = choices
                    quiz_temp[f"{number}_correct_answer"] = correct_answer
                    number += 1
                break
            except:
                err_msg = traceback.format_exc()
                print("2", err_msg)
                notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: quiz fail!!"
                post_slack(notification)
                break
        # print(story_json)
        # print(quiz_temp)
        try:
            temp = table_story_quiz.put_item(Item=quiz_temp)
        except:
            print("dynamodb query error!!")

        # sqs 전송 (SQS_post_midjourney_story)
        try:
            # 최종 처리는 sqs에 연결된 lambda가 진행
            sqs = boto3.resource("sqs", region_name="ap-northeast-2")
            queue = sqs.get_queue_by_name(QueueName=f"SQS_gpt_validation_{env}")
            temp_json = {}
            temp_json["user_id"] = user_id
            temp_json["time_stamp"] = time_stamp

            message_body = json.dumps(temp_json)
            response = queue.send_message(
                MessageBody=message_body,
            )

        except ClientError as error:
            logger.exception("Send Upscale message failed: %s", message_body)
            raise error

        print("good")
        return {"statusCode": 200, "body": "gg"}

    # MVP 2차 버전
    elif version == "2":
        try:
            env = "prod"
            # print(user_id,time_stamp)
            query = f"SELECT * FROM Dy_story_event_{env} where user_id='{user_id}' and time_stamp='{time_stamp}'"
            # print(query)
            result = dynamodb_client.execute_statement(Statement=query)

            mode = result["Items"][0]["mode"]["S"]
        except:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: event_parsing_fail!"
            post_slack(notification)
            print("event_parsing_fail!!!!!")
            return {"statusCode": 200, "body": "..."}

        age = str(result["Items"][0]["age"]["S"])
        gender = result["Items"][0]["gender"]["S"]
        name = result["Items"][0]["name"]["S"]

        # 주제 추천
        if mode == "0":
            major = result["Items"][0]["major"]["S"]
            middle = result["Items"][0]["middle"]["S"]
            sub = result["Items"][0]["sub"]["S"]
            message = [
                {
                    "role": "user",
                    "content": "너는 아이들을 위한 동화 생성하고 이야기해 주는 작가 선생님으로, 아이들에게 이야기해주는 톤을 유지하며, 아래의 캐릭터설정을 기반으로 동화분류와 제약사항을 참고하여 그 구성과 아래 출력형식(JSON)에 맞춰 동화를 작성해라.\n\n동화분류:\n대분류: %(major)s;\n중분류:%(middle)s;\n소분류:%(sub)s\n\n제약사항:\n최대페이지수: 8;\n페이지수: 8;\n페이지의양: 최소 30 tokens;\n제목:캐릭터의 이름이 들어가는 동화제목\n대상: 3~6세\n말투: 했어요;\n\n캐릭터설정:\n이름:%(name)s;\n나이:%(age)s살,\n성별:%(gender)s\n\n출력형식:{제목:동화의 제목, 1:페이지 내용, 2:페이지 내용, 3:페이지 내용, 4:페이지 내용, 5:페이지 내용, 6:페이지 내용, 7:페이지 내용, 8:페이지 내용}"
                    % {
                        "major": major,
                        "middle": middle,
                        "sub": sub,
                        "name": name,
                        "age": age,
                        "gender": gender,
                    },
                }
            ]

        # 유저 직접 입력
        elif mode == "1":
            background = result["Items"][0]["background"]["S"]
            theme = result["Items"][0]["theme"]["S"]
            message = [
                {
                    "role": "user",
                    "content": "너는 아이들을 위한 동화 생성하고 이야기해 주는 작가 선생님으로, 아이들에게 이야기해주는 톤을 유지하며, 아래의 캐릭터설정을 기반으로 동화설정과 제약사항을 참고하여 그 구성과 아래 출력형식(JSON)에 맞춰 동화를 작성해라.\n\n동화분류:\n배경:%(background)s;\n상황:%(theme)s;\n\n제약사항:\n최대페이지수: 8;\n페이지수: 8;\n페이지의양: 최소 30 tokens;\n제목:캐릭터의 이름이 들어가는 동화제목\n대상: 3~6세\n말투: 했어요. \n\n캐릭터설정:\n이름:%(name)s;\n나이:%(age)s살,\n성별:%(gender)s\n\n출력형식:{제목:동화의 제목, 1:페이지 내용, 2:페이지 내용, 3:페이지 내용, 4:페이지 내용, 5:페이지 내용, 6:페이지 내용, 7:페이지 내용, 8:페이지 내용}"
                    % {
                        "background": background,
                        "theme": theme,
                        "name": name,
                        "age": age,
                        "gender": gender,
                    },
                }
            ]

        # json 형식 검사
        temperature = 1
        flag = False
        while temperature > 0.7:
            try:
                story = make_stroy_gpt_4(message, temperature=temperature)
                print(story)
                story_json = json.loads(story)
                if len(list(story_json.keys())) == 9:
                    title = story_json[list(story_json.keys())[0]]
                    print(title)
                    del story_json[list(story_json.keys())[0]]

                    f = True
                    # gpt에서 글이 생성되지 않는 경우 방지...
                    for key in list(story_json.keys()):
                        if len(story_json[key]) == 0:
                            f = False
                    if f:
                        flag = True
                        break
                else:
                    temperature -= 0.1
            except:
                try:
                    check = story.split("{")
                    if len(check[0]) > 1:
                        if len(check) == 2:
                            temp_story = "{" + check[1]
                            print(temp_story)
                            story_json = json.loads(temp_story)
                            try:
                                if len(list(story_json.keys())) == 9:
                                    title = story_json[list(story_json.keys())[0]]
                                    print(title)
                                    del story_json[list(story_json.keys())[0]]

                                    f = True
                                    # gpt에서 글이 생성되지 않는 경우 방지...
                                    for key in list(story_json.keys()):
                                        if len(story_json[key]) == 0:
                                            f = False
                                    if f:
                                        flag = True
                                        break
                            except:
                                pass
                except:
                    print("haha")

                temperature -= 0.1
                err_msg = traceback.format_exc()
                print("1", err_msg)

        if not flag:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: gpt_fail!!"
            post_slack(notification)
            print("gpt fail!!!")
            return {"statusCode": 200, "body": "gg"}

        try:
            story_image = {}

            story_josn_key = list(story_json.keys())

            for key_index in range(len(story_josn_key)):
                print(story_json[story_josn_key[key_index]])

                message_image = [
                    {
                        "role": "system",
                        "content": f"너는 주어진 동화내용에 대한 삽화를 생성하는 삽화가이다. 주어진 동화내용을 이해한 후 이에 어울리는 장면을 영어명사구로 아래 제약사항에 맞춰 출력해라.\n\n제약사항:\n영어로 작성한다.\n반드시 캐릭터 이름을 성별에 따른 인칭대명사로 치환한다.\nboy, girl, man, women과 같은 캐릭터를 지칭하는 단어를 사용하지 않는다.\n등장인물과 주변 객체 간의 행위에 초점을 둔다.\n고유명사를 제거한다.\n캐릭터의 행동을 묘사한다.\n절대 고유명사를 사용하지 않는다.\n반드시 캐릭터는 인간이다.\n절대 출력에 고유명사를 사용하지 않는다.\n출력에는 캐릭터의 이름이 등장하지 않는다.\n장면에 대한 최대한 구체적인 묘사를 출력한다.\n\n캐릭터:\n이름:{name};\n나이:{age}살;\n성별:{gender};",
                    },
                    {
                        "role": "user",
                        "content": f"동화내용:\n어느 날, 꼬미가 밤하늘에 뜬 거대한 보름달을 보고 있었어요. '우와, 보름달이 정말 크고 둥글다!' 꼬미는 감탄했어요.\n출력:\ninside a under the moonlight, happy, looking at moon, pointing the moon, astonishment\n동화내용:\n어머낫!!, 날 볼 수 있나봐!' 토끼가 대답했어요. '나는 달토끼라고 해. 반가워 꼬미야!\n출력:\ninside a moon surface, a white cute rabbit, the white rabbit shakes his hand, happy\n동화내용:\n그러다가 꼬미는 흐릿한 지구를 보았어요. '우리 집은 저기야!' 꼬미는 달토끼에게 자신의 이야기를 들려주었어요.\n출력:\ninside a moon surface, girl and a white cute rabbit pointing at the blurry Earth\n동화내용:\n작은 강아지 의사는 여수가 진료실에 들어오자 기뻐했어요. 그래서 강아지 의사는 여수에게 하루 종일 어떻게 일하는 지 보여주기로 했어요.\n출력:\ninside a hospital, A little puppy doctor, happy, showing something to a young girl in the clinic\n동화내용:\n'북'이 다시 새에게 돌아가서 섬세하게 날개를 진료했어요. 무언가 부서진 것을 찾아 고쳐주었답니다.\n출력:\ninside a hospital,  touches the small bird with hands.\n동화내용:\n그래서 아이들, 우리도 환경구 군처럼 우리 환경을 사랑하고, 쓰레기는 쓰레기통에 버리는 걸 잊지 말아요. 안녕!\n출력:\ninside a garbage dump, smiling at the garbage dump.\n동화내용:\n{story_json[story_josn_key[key_index]]};\n출력:",
                    },
                ]

                # 무한루프에 빠지는 경우 존재
                cnt = 0
                while cnt < 3:
                    try:
                        image_prompt = make_stroy(message_image, temperature=1)

                        # 한글이 존재하면 최대 세번까지 다시 생성하도록
                        temp = image_prompt
                        word_check = re.sub(r"[^ㄱ-ㅣ가-힣\s]", "", temp)
                        word_check = word_check.replace(" ", "")

                        # 한국어가 생성된 것
                        if len(word_check) != 0:
                            cnt += 1
                            time.sleep(1)
                            print(image_prompt)
                            continue
                        break
                    except:
                        err_msg = traceback.format_exc()
                        print("2", err_msg)

                # 문제가 발생한 것
                if cnt == 3:
                    notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: 이미지 프롬프트 생성 이슈 (한국어 존재)!!"
                    post_slack(notification)

                story_image[story_josn_key[key_index]] = image_prompt
                print(image_prompt)

        except:
            err_msg = traceback.format_exc()
            print("3", err_msg)
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: check plz!!"
            post_slack(notification)
            print("something went wrong!!")
            return {"statusCode": 500, "body": "chatgpt error!!"}

        dynamodb = boto3.resource("dynamodb", region_name="ap-northeast-2")
        table_gpt_story = dynamodb.Table(f"Dy_gpt_story_{env}")
        table_gpt_prompt = dynamodb.Table(f"Dy_gpt_prompt_{env}")
        table_story_summary = dynamodb.Table(f"Dy_story_summary_prod")
        try:
            # 스토리
            story_json["user_id"] = user_id
            story_json["time_stamp"] = time_stamp
            temp = table_gpt_story.put_item(Item=story_json)

            # 스토리 상황묘사
            story_image["user_id"] = user_id
            story_image["time_stamp"] = time_stamp
            temp = table_gpt_prompt.put_item(Item=story_image)

            # 책 제목 설정
            query = f"UPDATE Dy_story_event_{env} SET title = '{title}' WHERE user_id='{user_id}' and time_stamp='{time_stamp}';"
            result = dynamodb_client.execute_statement(Statement=query)

        except:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: dynamodb error!"
            post_slack(notification)
            print("put dynamodb error!!!!")

        # 줄거리 생성
        try:
            story = ""
            for key_index in range(len(story_josn_key)):
                story += " " + story_json[story_josn_key[key_index]]

            message = [
                {
                    "role": "user",
                    "content": "다음 동화를 이해하고 해당 동화에 대한 줄거리를 두 문장으로 작성해줘.\n\n동화:\n%(story)s;"
                    % {
                        "story": story,
                    },
                }
            ]
            while True:
                try:
                    story_summary = make_stroy(message, temperature=1)
                    break
                except:
                    err_msg = traceback.format_exc()
                    print("2", err_msg)

            print(story_summary)
            story_summary_json = {}
            story_summary_json["user_id"] = user_id
            story_summary_json["time_stamp"] = time_stamp
            story_summary_json["summary"] = story_summary
            temp = table_story_summary.put_item(Item=story_summary_json)

        except:
            print("error occur!!")

        # 퀴즈 생성
        table_story_quiz = dynamodb.Table(f"Dy_story_quiz_prod")

        message = [
            {
                "role": "user",
                "content": "너는 주어진 동화에 대한 퀴즈를 만들어주는 작가 선생님으로, 아이들에게 이야기해주는 톤을 유지하며, 아래의 동화제목, 동화내용을 기반으로 제약사항을 참고하여 아래 출력형식(JSON)에 맞춰 퀴즈를 생성해라.\n\n동화제목: %(title)s\n동화내용: %(story)s\n\n제약사항:\n최대 question 개수: 4개;\nquestion 개수: 4개;\nchoices수: 4개;\n출력형식:{quiz_title:"
                ",quiz_questions:{[question:"
                ", choices:"
                ",correct_answer:"
                "},{question:"
                ", choices:"
                ",correct_answer:"
                "},{question:"
                ", choices:"
                ",correct_answer:"
                "},{question:"
                ", choices:"
                ",correct_answer:}]}"
                % {
                    "title": title,
                    "story": story,
                },
            }
        ]
        quiz_temp = {}
        quiz_temp["user_id"] = user_id
        quiz_temp["time_stamp"] = time_stamp

        # 세 번 까지는 다시 생성하도록 하는 로직 추가
        cnt = 0
        while cnt < 3:
            try:
                story_quiz = make_stroy(message, temperature=1)
                story_json = json.loads(story_quiz)
                print(story_json["quiz_title"])
                print(story_json["quiz_questions"])
                number = 1
                for question_temp in story_json["quiz_questions"]:
                    question = question_temp["question"]
                    choices = question_temp["choices"]
                    correct_answer = question_temp["correct_answer"]
                    quiz_temp[f"{number}_question"] = question
                    quiz_temp[f"{number}_choices"] = choices
                    quiz_temp[f"{number}_correct_answer"] = correct_answer
                    number += 1
                break
            except:
                # err_msg = traceback.format_exc()
                # print("2", err_msg)
                cnt += 1

        if cnt == 3:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: quiz fail!!"
            post_slack(notification)

        try:
            temp = table_story_quiz.put_item(Item=quiz_temp)
        except:
            print("dynamodb query error!!")

        # 영어버전 생성 << text에 대해서만 진행
        try:
            query = f"select * from Dy_gpt_story_prod where user_id='{user_id}' and time_stamp='{time_stamp}'"
            story_result = dynamodb_client.execute_statement(Statement=query)

            story = ""
            story += f"title:{title}\n"

            for i in range(1, 9):
                temp = story_result["Items"][0][str(i)]["S"]
                story += f"{i}:{temp}\n"

            print(story)
            message = [
                {
                    "role": "user",
                    "content": "너는 아이들을 위한 동화를 번역하는 번역가로 한국어로 작성된 다음의 동화 내용을 이해하고 해당 내용을 영어로 제약조건과 출력형식(JSON)을 반드시 지켜 번역해줘.\n동화내용:\n%(story)s\n제약조건:\n3세 유아가 이해할 수 있는 어휘를 사용한다.\n새로운 내용을 추가하지 않는다.\n기존의 내용을 그대로 영어로 번역한다.\n\n출력형식:\n{title:동화의 제목, 1:페이지 내용, 2:페이지 내용, 3:페이지 내용, 4:페이지 내용, 5:페이지 내용, 6:페이지 내용, 7:페이지 내용, 8:페이지 내용}"
                    % {
                        "story": story,
                    },
                }
            ]

            cnt = 0
            while cnt < 3:
                try:
                    story_eng = make_stroy(message, temperature=1)
                    story_json = json.loads(story_eng)
                    print(story_json)
                    break
                except:
                    err_msg = traceback.format_exc()
                    print("2", err_msg)
                    # return {"statusCode": 200, "body": "why??"}
                    cnt += 1

            if cnt == 3:
                print("english put fail!")
                notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: english version fail!!"

            # DB put 작업진행
            try:
                story_json["user_id"] = user_id
                story_json["time_stamp"] = time_stamp
                table = dynamodb.Table(f"Dy_gpt_story_english_prod")
                temp = table.put_item(Item=story_json)
            except:
                print("english put fail!")
                # notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: english version fail!!"
                # post_slack(notification)
                print(story_eng)

        except:
            notification = f"from La_post_gpt: user_id:{user_id}, time_stamp:{time_stamp}, reason: english version fail!!"
            post_slack(notification)
            print("english version fail!")

        # sqs 전송 (SQS_post_midjourney_story)
        try:
            # 최종 처리는 sqs에 연결된 lambda가 진행
            sqs = boto3.resource("sqs", region_name="ap-northeast-2")
            queue = sqs.get_queue_by_name(QueueName=f"SQS_gpt_validation_{env}")
            temp_json = {}
            temp_json["user_id"] = user_id
            temp_json["time_stamp"] = time_stamp

            message_body = json.dumps(temp_json)
            response = queue.send_message(
                MessageBody=message_body,
            )

        except ClientError as error:
            logger.exception("Send Upscale message failed: %s", message_body)
            raise error

        print("good")
        return {"statusCode": 200, "body": "gg"}
