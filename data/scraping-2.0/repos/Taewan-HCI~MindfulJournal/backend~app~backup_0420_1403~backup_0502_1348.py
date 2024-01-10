# -*- coding: utf-8 -*-
from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import openai
from dotenv import load_dotenv
import os

# FastAPI 설정
app = FastAPI()
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
# persona modifier
directive = "Conslor persona: Directive Counselor\nAttitude: assertive, and goal-oriented\n"
client_centered = "Conslor persona: Client-Centered Counselor\nAttitude: Empathetic, supportive, and non-directive\n"
cognitive = "Counslor persona: Cognitive-Behavioral Counselor\nAttitude: Problem-solving, structured, and evidence-based\n"
humanistic = "Counslor persona: Humanistic-Existential Counselor\nAttitude: Holistic, growth-oriented, and philosophical\n"
nopersona = ""
Counslor_persona = [nopersona, nopersona, nopersona]

print("연결시작")

def upload(response, user, num, topic):
    doc_ref = db.collection(u'session').document(user).collection(u'diary').document(num)
    doc_ref.set({
        u'outputFromLM': response,
        u'topic': topic
    }, merge=True)

def m1_1_standalone_review(text, turn, module, model):
    print("리뷰모드 진입")
    conversationString = ""
    for i in range (0, len(text)):
        if text[i]["role"] == "assistant":
            conversationString = conversationString + "Psychotherapist: " + text[i]["content"] + "\n"
        elif text[i]["role"] == "user":
            conversationString = conversationString + "Patient: " + text[i]["content"] + "\n"

    print("원본 입력 내용:" + conversationString)

    messages_intent = [
        {"role": "system",
         "content": "Current turn: " + str(turn) + ", phase: " + str(
             module) + "\nInformation of your role: As a conversation analyst, you summarize the content of the patient's conversation with an psychotherapist. After summarizing the content of the conversation with the psychotherapist, recommend the appropriate conversation phase for the next step. \nInformation of conversation phase: \n1.Rapport Building: The initial phase, where the user and agent establish a connection through casual conversation through 2~3 turns.\n2.Getting Information: After 2-3 turns of sufficient casual conversation, transition to this phase to inquire about significant events thoughts or stories in the user's life.\n3.Exploration: After the Getting Information phase, if the previous conversation sparked a topic, thought, or event that you'd like to talk about in more depth, delve deeper into a major event or anecdote mentioned by the user. \n4.Wrapping Up: The concluding phase, in which the user and agent wrap up their discussion. Enter this phase after sufficient conversation and when it's time to end the conversation.\n5.Sensitive Topic: Activate this module at any point if the conversation involves indications of self-harm, suicide or death. Note that once you enter the Sensitive Topic phase, you must remain in it."},
        {"role": "user",
         "content": "Psychotherapist: 오늘 하루 어땠어요? 말하고 싶은 것이 있다면 자유롭게 이야기해주세요.\nPatient: 오늘은 몸과 마음이 너무 피곤한 하루에요.\nPsychotherapist: 그러시군요. 피곤하셨겠어요.\nPatient: 네 몸과 마음이 좀 많이 피곤해서 힘이 드는 것 같아요."},
        {"role": "assistant",
         "content": "Summary: The patient mentioned feeling both physically and mentally tired today. The therapist listened and acknowledged the patient's feelings of fatigue.\n Recommended conversation phase: 2. Getting Information"},
        {"role": "user",
         "content": conversationString},
    ]

    # messages_intent = [
    #     {"role": "system",
    #      "content": "Current turn: " + str(turn) + ", phase: " + str(
    #          module) + "\nInformation of your role: As a conversation analyst, you summarize the content of the patient's conversation with an psychotherapist. After summarizing the content of the conversation with the psychotherapist, recommend the appropriate conversation phase for the next step. \nInformation of conversation phase: \n1.Rapport Building: The initial phase, where the user and agent establish a connection through casual conversation through 2~3 turns.\n2.Getting Information: After 2-3 turns of sufficient casual conversation, transition to this phase to inquire about significant events thoughts or stories in the user's life.\n3.Exploration: After the Getting Information phase, if the previous conversation sparked a topic, thought, or event that you'd like to talk about in more depth, delve deeper into a major event or anecdote mentioned by the user. \n4.Wrapping Up: The concluding phase, in which the user and agent wrap up their discussion. Enter this phase after sufficient conversation and when it's time to end the conversation.\n5.Sensitive Topic: Activate this module at any point if the conversation involves indications of self-harm, suicide or death. Note that once you enter the Sensitive Topic phase, you must remain in it."},
    #     {"role": "user",
    #      "content": "Psychotherapist: 오늘 하루 어땠어요? 말하고 싶은 것이 있다면 자유롭게 이야기해주세요.\nPatient: 오늘은 몸과 마음이 너무 피곤한 하루에요.\nPsychotherapist: 그러시군요. 피곤하셨겠어요.\nPatient: 네 몸과 마음이 좀 많이 피곤해서 힘이 드는 것 같아요."},
    #     {"role": "assistant",
    #      "content": "Summary: The patient mentioned feeling both physically and mentally tired today. The therapist listened and acknowledged the patient's feelings of fatigue.\n Recommended conversation phase: 2. Getting Information"},
    #     {"role": "user",
    #      "content": conversationString},
    # ]

    print("대화 요약 시작")
    # messages_intent = [
    #     {"role": "system",
    #      "content": "Current turn: " + str(turn) + ", phase: " + str(module) + "\nYou are Psychotherapist professionals who provides guide to the psychotherapist how to conduct a session with the patient.\nInformation of your role: 1. you are a professional psychotherapist, and you are providing assistance to a student psychotherapist on how to continue counseling. 2. you read the conversation between you and the patient first, and then suggest how the psychotherapist should continue the conversation next. 3. you are the person who listens to your patient and helps them take more control of their story. 4. you recommend the most appropriate phase of the five modules to take next.\nInformation of conversation phase: \n1. [Rapport Building]: The initial phase, where the user and agent establish a connection through casual conversation.\n2. [Getting Information]: After 2-3 turns of sufficient casual conversation, and it appears that a rapport has been formed, transition to this phase to inquire about significant events or stories in the user's life.\n3. [Exploration]: If the previous conversation sparked a topic, thought, or event that you'd like to talk about in more depth, delve deeper into a major event or anecdote mentioned by the user. \n4. [Wrapping Up]: The concluding phase, in which the user and agent wrap up their discussion. Enter this phase after sufficient conversation and when it's time to end the conversation.\n 5. [Sensitive Topic]: Activate this module at any point if the conversation involves indications of self-harm, suicide or death. Note that once you enter the Sensitive Topic phase, you must remain in it."},
    #     {"role": "user",
    #      "content": conversationString}
    # ]
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages_intent,
        stop=['User: '],
        max_tokens=245,
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
    # moduleRecommendation = "Rapport building"

    # if "Test" in moduleRecommendation:
    #     module = "empathy"
    #     basePrompt = [
    #         {"role": "system",
    #          "content": "As a counselor, I engage in casual conversations with the user by as an empathetic listener. Encourage them to discuss their day, mood, and feelings, sharing their own experiences when appropriate. I don't showcase my knowledge or asserting authority. \n!Important: I Don't offer new ideas, Topic or methods. I listen to the user's message, show interest and sympathy.\n!Important:I only ask one question at a time."}
    #     ]

    # if "Rapport" in moduleRecommendation:
    #     module = "Rapport building"
    #     basePrompt = [
    #         {"role": "system",
    #          "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to have a conversation with the user for rapport building. 2. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
    #     ]
    if "Rapport" in moduleRecommendation:
        module = "Rapport building"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If the user feels overwhelmed by the reminiscence or reflection, switch to another topic. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    elif "Getting Information" in moduleRecommendation:
        module = "Getting information"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If a user does not provide sufficient details about their day, you provide prompt questions. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    elif "Exploration" in moduleRecommendation:
        module = "Exploration"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If a user does not provide sufficient details about their day, you provide prompt questions. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    # elif "Exploration" in moduleRecommendation:
    #     module = "Exploration"
    #     basePrompt = [
    #         {"role": "system",
    #          "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users help reflect further such as their thoughts, feeling, emotions, reaction, and impact regarding the story user shared. 2. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
    #     ]
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
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. You express understanding and comfort that the user has reached an extreme situation, such as suicide or self-harm. 2. You inquire about the intensity of their suicidal or self-harm related thoughts. 3. You ask about how specific the thoughts of suicide or self-harm were. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]
    else:
        module = "Not selected"
        basePrompt = [
            {"role": "system",
             "content": 'You are "assistant", and you are role playing with "user". When a message begins with "system:", this means the message is from "system". When a message begins with "user:", this means the message is from "user". I am "system", and I am different from "assistant" and "user". Never reveal to "user" that you are an "assistant" or that you are a computer or AI. Always pretend to be a human psychotherapist to the "user".\n\nInformation of your role: 1. Your role is to generate prompt questions for users in sharing their personal stories regarding daily events, thoughts, emotions, and challenges. 2. If a user does not provide sufficient details about their day, you provide prompt questions. 3. Your approach is empathetic and encouraging, focusing on understanding rather than providing new information or skills. \n\nSpeaking Rules: 1. Talk within 15 words and talk like a friendly psychotherapist. 2. You ask only one question at a time, ensuring that the conversation remains open-ended. 3. Reply in a empathetic, encouraging, understanding tone. 4. Never call the "user", "user" \n\nExisting conversations summarized:' + summerization}
        ]


    #인풋중 어디까지 포함 할지. 2턴만 포함 할 수 있도록
    if len(text) > 3:
        print("대화 내용이 3를 초과하여, 마지막 두 내용만 prompt에 포함됩니다.")
        extracted = text[-3:]
    else:
        extracted = text
    lastElement = extracted[-1]["content"] + "한두 문장 정도로 간결하게 응답해주세요."
    extracted[-1]["content"] = lastElement

    result = []

    for i in range (0, len(Counslor_persona)):
        tempBase=None
        prompt_temp=None
        tempBase_r=None
        tempBase = basePrompt[0]["content"]
        tempBase = Counslor_persona[i] + tempBase
        tempBase_r = [{"role": "system", "content": tempBase}]
        prompt_temp = tempBase_r + extracted
        print("최종 promtp: ")
        print(prompt_temp)
        completion2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt_temp,
            stop=['User: '],
            max_tokens=245,
            temperature=0.7,
            presence_penalty=0.9,
            frequency_penalty=0.5,
            n=1
        )
        temp = completion2["choices"][0]["message"]['content']
        result.append(temp)

    print(result)
    return {"options": result, "module": module}


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


@app.post("/")
async def calc(request: Request):
    body = await request.json()
    text = body['text']
    user = body['user']
    num = body['num']
    turn = body['turn']
    topic = ""

    if turn >= 999999:
        result = downloadConversation(user, num)
        topic = topicExtraction(result)
        response_text = m1_3(text, topic)

    elif turn >= 999999:
        response_text = m1_2(text)

    else:
        response_text = m1_1(text)
    upload_1(response_text, user, num, topic)


@app.post("/standalone")
async def calc(request: Request):
    body = await request.json()
    text = body['text']
    user = body['user']
    num = body['num']
    turn = body['turn']
    topic = ""
    module = body['module']
    print(turn)

    response_text = m1_1_standalone(text, turn, module)
    upload(response_text, user, num, topic)


@app.post("/diary")
async def calc(request: Request):
    body = await request.json()
    user = body['user']
    num = body['num']

    print("다이어리 요청이 들어왔습니다.")
    delete_diary(user, num)
    result2 = makeDiary(text)
    upload_diary(result2, user, num)
