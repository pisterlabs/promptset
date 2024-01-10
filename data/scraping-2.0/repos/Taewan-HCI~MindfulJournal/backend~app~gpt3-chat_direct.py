# -*- coding: utf-8 -*-
from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import openai
import time
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# FastAPI 설정
app = FastAPI()s
origins = ["*", "http://localhost:3000", "https://mindful-journal-frontend-s8zk.vercel.app/"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 구글 Firebase인증 및 openAI api key 설정
load_dotenv()
gptapi = os.getenv("gptkey")
cred = credentials.Certificate(
    '/Users/taewankim/PycharmProjects/LLM_diary/backend/mindfuljournal-44166-firebase-adminsdk-frpg8-10b50844cf.json')
app_1 = firebase_admin.initialize_app(cred)
db = firestore.client()
My_OpenAI_key = gptapi
openai.api_key = My_OpenAI_key

# Prompt 관련 자료

print("연결시작")

# persona modifier
directive = "Conslor persona: Directive Counselor\nAttitude: assertive, and goal-oriented\n"
client_centered = "Conslor persona: Client-Centered Counselor\nAttitude: Empathetic, supportive, and non-directive\n"
cognitive = "Counslor persona: Cognitive-Behavioral Counselor\nAttitude: Problem-solving, structured, and evidence-based\n"
humanistic = "Counslor persona: Humanistic-Existential Counselor\nAttitude: Holistic, growth-oriented, and philosophical\n"
nopersona = ""
Counslor_persona = [nopersona, nopersona, nopersona]


class EmailSchema(BaseModel):
    to: str
    subject: str
    body: str


def upload(response, user, num, topic):
    doc_ref = db.collection(u'session').document(user).collection(u'diary').document(num)
    doc_ref.set({
        u'outputFromLM': response,
        u'topic': topic
    }, merge=True)


def download():
    doc_ref = db.collection(u'session').document("ut01@test.com").collection(u'diary').document("G02")
    doc = doc_ref.get()
    if doc.exists:
        print(f'Document data: {doc.to_dict()}')
        return doc.to_dict()
    else:
        print(u'No such document!')


def upload_operator(response, user, num, topic):
    doc_ref = db.collection(u'session').document(user).collection(u'diary').document(num)

    doc = doc_ref.get()
    if doc.exists:
        doc_ref.update({
            u'outputForReview': response,
            u'history_serverside': firestore.ArrayUnion([{'response': response}]),
            u'status': "new",
        })
    else:
        doc_ref.set({
            u'outputForReview': response,
            u'history_serverside': [{'response': response}],
            u'status': "new",
        })

    doc_ref.set({
        u'outputForReview': response,
        u'status': "new",
        u'topic': topic
    }, merge=True)


def upload_diary(response, user, num):
    doc_ref = db.collection(u'session').document(user).collection(u'diary').document(num)
    doc_ref.set({
        u'diary': response
    }, merge=True)


def m1_1_standalone_review(text, turn, module, model):
    print("리뷰모드 진입")
    conversationString = ""
    for i in range(0, len(text)):
        if text[i]["role"] == "assistant":
            conversationString = conversationString + "Psychotherapist: " + text[i]["content"] + "\n"
        elif text[i]["role"] == "user":
            conversationString = conversationString + "Patient: " + text[i]["content"] + "\n"

    print("원본 입력 내용:" + conversationString)

    messages_intent = [
        {"role": "system",
         "content": "Current turn: " + str(turn) + ", phase: " + str(
             module) + "\nInformation of your role: As a conversation analyst, you summarize the content of the patient's conversation with an psychotherapist. After summarizing the content of the conversation with the psychotherapist, recommend the appropriate conversation phase for the next step. \nInformation of conversation phase: \n1.Main session: Continue the conversation in the 'Main session' unless the user expresses a desire to end the conversation, or sensitive topics (self-harm, suicide). \n2.Wrapping Up: If the user has expressed a desire to end the conversation, you suggest a 'Wrapping Up' phase.\n3.Sensitive Topic: Activate this module at any point if the user expressed indications of self-harm, suicide or death. \n Rule: If there is no mention of suicide or self-harm, You can not recommend the 'Sensitive Topic'."},
        {"role": "user",
         "content": "Psychotherapist: 오늘 하루 어땠어요? 말하고 싶은 것이 있다면 자유롭게 이야기해주세요.\nPatient: 오늘은 몸과 마음이 너무 피곤한 하루에요.\nPsychotherapist: 그러시군요. 피곤하셨겠어요.\nPatient: 네 몸과 마음이 좀 많이 피곤해서 힘이 드는 것 같아요."},
        {"role": "assistant",
         "content": "Summary: The patient mentioned feeling both physically and mentally tired today. The therapist listened and acknowledged the patient's feelings of fatigue.\n Recommended conversation phase: 2. Getting Information"},
        {"role": "user",
         "content": conversationString},
    ]

    print("대화 요약 시작")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_intent,
        stop=['User: '],
        max_tokens=2048,
        temperature=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.5,
    )
    result_1 = completion["choices"][0]["message"]['content']
    print("모듈 추천 내용: " + result_1)
    split_strings = result_1.split("\n")
    print(split_strings)
    temp = split_strings[0]
    split_strings_2 = temp.split(": ")
    summerization = split_strings_2[-1]
    moduleRecommendation = split_strings[-1]

    if "Main" in moduleRecommendation:
        module = "Main session"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If the user feels overwhelmed by the reminiscence or reflection, switch to another topic. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]

    elif "Wrapping" in moduleRecommendation:
        module = "Wrapping up"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to close the conversation after ensuring that users have no additional topics to discuss. 2. You adopt a supportive and empathetic approach, asking if user have any remaining concerns or thoughts they would like to share. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    elif "Sensitive" in moduleRecommendation:
        module = "Sensitive"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. First, you empathize with the struggle and offer a comforting message. 2. You inquire about the intensity of their suicidal or self-harm related thoughts. 3. You ask about how specific the thoughts of suicide or self-harm were. \n\nSpeaking Rules: 1. First, you empathize with the struggle and offer a comforting message. 2. Talk within 15 words and talk like a friendly psychotherapist. 3. You ask only one question at a time. 4. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    else:
        module = "Not selected"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If a user does not provide sufficient details about their day, you provide prompt questions. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]

    # 인풋중 어디까지 포함 할지. 2턴만 포함 할 수 있도록
    if len(text) > 3:
        print("대화 내용이 3를 초과하여, 마지막 두 내용만 prompt에 포함됩니다.")
        extracted = text[-3:]
    else:
        extracted = text
    lastElement = extracted[-1]["content"] + "한두 문장 정도로 간결하게 응답해주세요."
    extracted[-1]["content"] = lastElement

    result = []

    tempBase = None
    prompt_temp = None
    tempBase_r = None
    tempBase = basePrompt[0]["content"]
    tempBase_r = [{"role": "system", "content": tempBase}]
    prompt_temp = tempBase_r + extracted
    print("최종 promtp: ")
    print(prompt_temp)
    completion2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_temp,
        stop=['User: '],
        max_tokens=2048,
        temperature=0.7,
        presence_penalty=0.9,
        frequency_penalty=0.5,
        n=4
    )
    print(completion2)
    for i in range(0, 4):
        result.append(completion2["choices"][i]["message"]['content'])
    print(result)
    return {"options": result, "module": module, "summary": summerization}

def diary(text):
    print("다이어리 시작")
    conversationString = ""
    test = ""
    for i in range(0, len(text)):
        if text[i]["role"] == "assistant":
            test = test + "Psychotherapist: " + text[i]["content"] + "\n"
        elif text[i]["role"] == "user":
            conversationString = conversationString + "Patient: " + text[i]["content"] + "\n"
    print(conversationString)
    prompt_for_diary = [{"role": "system",
                         "content": "I summarise the dialogue below in the form of a diary entry from the patient perspective. Summarise the events, feelings, and anecdotes from the conversation like a diary entry as you reflect on your day. I only organize my diary entries by what the patient mentions. Don't create diary entries for anything the patient didn't say."},
                        {"role": "user",
                         "content": "Psychotherapist: 오늘은 어떤 일이 있었나요? 정해진 규칙이 없으니 자유롭게 얘기해주세요.\nPatient: 오늘도 평범하게 출근하고, 뭐 별일은 없었어요.\nPsychotherapist: 그랬군요. 조금 더 자세한 이야기를 듣고 싶은데. 오늘 하루 가장 기억에 남는 한 가지 일을 골라본다면 무엇이 있을까요?\nPatient: 글쎄. 일단 오늘 아침에 업무 미팅이 하나 있었고, 저녁에는 오랜만에 부모님과 식사했네요.\nPsychotherapist: 그랬군요. 얘기해줘서 고마워요! 오늘 업무 미팅에서 별일은 없었나요?\nPatient: 음 큰 미팅은 아니었는데, 요즘 계속 제 상사가 저를 무시하고, 안 좋게 보고 있는 것 같다는 느낌이 들어서 힘들어요.\nPsychotherapist: 얘기해줘서 고마워요. 업무 미팅과 관련해서 어떤 감정이나 기분이 들었나요?\nPatient: 그냥 기분이 좋지 않고, 짜증나고, 앞으로 계속 일해야 하는 곳인데, 어떻게 계속 다녀야 하나 싶지. 약간 이 사람을 만날때마다 불편하기도 하고.\nPsychotherapist: 혹시 그렇게 느끼게 된 이유나 사건이 있을까요?\nPatient: 사실 나는 예전과 같이 똑같이 하고 있다고 생각하는데, 내가 메일을 보내면 답을 안하기도 부지기수이고, 뭔가 나를 무시하고 있다는 느낌을 계속 받는 것 같아. 그냥 뭔가 눈빛에 그런 느낌이 든다고 해야 하나.\nPsychotherapist: 나라면 더 힘들어했을 것 같아. 혹시 이후로 행동이나 태도에 변화가 있었나요?\nPatient: 사실 나도 잘 확신이 안가고 그러니, 눈치를 엄청 보게 되는 것 같아. 계속 왠만하면 웃으면서 대답하고, 항상 기분을 살피고, 상사에게 조금 쫄아있다는 느낌이 들 정도로."},
                        {"role": "assistant",
                         "content": "오늘의 일기: 오늘은 어제와 다를것이 없는 평범한 하루였다. 아침에 업무 미팅이 있었고, 부모님과 오랜만에 식사를 했다. 요즘 회사에서 상사가 나를 무시하고 안좋게 보고 있다는 느낌이 들어서 힘들다. 매일 마주치는 사람에게 그런 느낌을 받으니, 여기를 계속 다녀야 할지 고민이 되고 너무 불편하고 힘이든다. 나는 예전처럼 똑같이 행동하는 것 같은데, 상사가 나를 대하는 태도와 시선은 많이 달라진 것 같다. 그래서 요즘은 눈치를 많이 보는 것 같다. 왠만하면 억지로라도 웃으면서 대답하려하고. 쉽지 않은 것 같다."},
                        {"role": "user",
                         "content": conversationString}]

    completion_3 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=prompt_for_diary,
        stop=['Patient: ', 'Psychotherapist: '],
        max_tokens=4000,
        temperature=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.5
    )
    diary_1 = completion_3["choices"][0]["message"]['content']
    return diary_1


def m1_1_standalone(text, turn, module, model):
    print("리뷰모드 진입")
    conversationString = ""
    for i in range(0, len(text)):
        if text[i]["role"] == "assistant":
            conversationString = conversationString + "Psychotherapist: " + text[i]["content"] + "\n"
        elif text[i]["role"] == "user":
            conversationString = conversationString + "Patient: " + text[i]["content"] + "\n"

    print("원본 입력 내용:" + conversationString)

    messages_intent = [
        {"role": "system",
         "content": "Current turn: " + str(turn) + ", phase: " + str(
             module) + "\nInformation of your role: As a conversation analyst, you summarize the content of the patient's conversation with an psychotherapist. After summarizing the content of the conversation with the psychotherapist, recommend the appropriate conversation phase for the next step. \nInformation of conversation phase: \n1.Main session: Continue the conversation in the 'Main session' unless the user expresses a desire to end the conversation, or sensitive topics (self-harm, suicide). \n2.Wrapping Up: If the user has expressed a desire to end the conversation, you suggest a 'Wrapping Up' phase.\n3.Sensitive Topic: Activate this module at any point if the user expressed indications of self-harm, suicide or death. \n Rule: If there is no mention of suicide or self-harm, You can not recommend the 'Sensitive Topic'."},
        {"role": "user",
         "content": "Psychotherapist: 오늘 하루 어땠어요? 말하고 싶은 것이 있다면 자유롭게 이야기해주세요.\nPatient: 오늘은 몸과 마음이 너무 피곤한 하루에요.\nPsychotherapist: 그러시군요. 피곤하셨겠어요.\nPatient: 네 몸과 마음이 좀 많이 피곤해서 힘이 드는 것 같아요."},
        {"role": "assistant",
         "content": "Summary: The patient mentioned feeling both physically and mentally tired today. The therapist listened and acknowledged the patient's feelings of fatigue.\n Recommended conversation phase: 2. Getting Information"},
        {"role": "user",
         "content": conversationString},
    ]
    print("대화 요약 시작")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages_intent,
        stop=['User: '],
        max_tokens=2048,
        temperature=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.5,
    )
    result_1 = completion["choices"][0]["message"]['content']
    print("모듈 추천 내용: " + result_1)
    split_strings = result_1.split("\n")
    print(split_strings)
    temp = split_strings[0]
    split_strings_2 = temp.split(": ")
    summerization = split_strings_2[-1]
    moduleRecommendation = split_strings[-1]

    time.sleep(0.5)

    if "Main" in moduleRecommendation:
        module = "Main session"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If the user feels overwhelmed by the reminiscence or reflection, switch to another topic. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]

    elif "Wrapping" in moduleRecommendation:
        module = "Wrapping up"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to close the conversation after ensuring that users have no additional topics to discuss. 2. You adopt a supportive and empathetic approach, asking if user have any remaining concerns or thoughts they would like to share. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    elif "Sensitive" in moduleRecommendation:
        module = "Sensitive"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. First, you empathize with the struggle and offer a comforting message. 2. You inquire about the intensity of their suicidal or self-harm related thoughts. 3. You ask about how specific the thoughts of suicide or self-harm were. \n\nSpeaking Rules: 1. First, you empathize with the struggle and offer a comforting message. 2. Talk within 15 words and talk like a friendly psychotherapist. 3. You ask only one question at a time. 4. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    else:
        module = "Not selected"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If a user does not provide sufficient details about their day, you provide prompt questions. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]

    # 인풋중 어디까지 포함 할지. 2턴만 포함 할 수 있도록
    if len(text) > 8:
        print("대화 내용이 3를 초과하여, 마지막 두 내용만 prompt에 포함됩니다.")
        extracted = text[-3:]
    else:
        extracted = text
    lastElement = extracted[-1]["content"] + "한두 문장 정도로 간결하게 응답해주세요."
    extracted[-1]["content"] = lastElement

    result = []

    tempBase = basePrompt[0]["content"]
    tempBase_r = [{"role": "system", "content": tempBase}]
    prompt_temp = tempBase_r + extracted
    print("최종 promtp: ")
    print(prompt_temp)
    completion2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=prompt_temp,
        stop=['User: '],
        max_tokens=2048,
        temperature=0.7,
        presence_penalty=0.9,
        frequency_penalty=0.5,
        n=1
    )
    print(completion2)
    for i in range(0, 1):
        result.append(completion2["choices"][i]["message"]['content'])

    print(result)
    return {"options": result, "module": module, "summary": summerization}


@app.get("/test", tags=["root"])
async def read_root() -> dict:
    response = download()
    return response
    # return {"message": "서버연결됨"}


@app.post("/review")
async def calc(request: Request):
    body = await request.json()
    text = body['text']
    user = body['user']
    num = body['num']
    turn = body['turn']
    topic = ""
    module = body['module']
    model = body['model']
    print(turn)

    response_text = m1_1_standalone_review(text, turn, module, model)
    upload(response_text, user, num, topic)


@app.post("/standalone")
async def calc(request: Request):
    body = await request.json()
    text = body['text']
    user = body['user']
    num = body['num']
    turn = body['turn']
    topic = ""
    module = body['module']
    model = body['model']
    print(turn)

    response_text = m1_1_standalone(text, turn, module, model)
    upload(response_text, user, num, topic)


@app.post("/operator")
async def calc(request: Request):
    body = await request.json()
    text = body['text']
    user = body['user']
    num = body['num']
    turn = body['turn']
    topic = ""
    module = body['module']
    model = body['model']
    print(turn)

    response_text = m1_1_standalone_review(text, turn, module, model)
    upload_operator(response_text, user, num, topic)


@app.post("/diary")
async def calc(request: Request):
    body = await request.json()
    text = body['text']
    user = body['user']
    num = body['num']

    response_text = diary(text)
    upload_diary(response_text, user, num)


@app.post("/send-email")
async def send_email(email: EmailSchema):
    message = Mail(
        from_email='twkim24@gmail.com',  # Change to your verified sender
        to_emails=email.to,
        subject=email.subject,
        plain_text_content=email.body)
    try:
        sg = SendGridAPIClient(
            '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')  # Replace with your SendGrid API Key
        response = sg.send(message)
        return {"message": "Email sent successfully"}
    except Exception as e:
        print(e.message)
        return {"message": "Failed to send email"}
