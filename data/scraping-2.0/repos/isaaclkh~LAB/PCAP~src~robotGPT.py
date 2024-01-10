import openai

from src.textFinal import text_to_speech, stt
import src.data.oled_list as oled

openai.api_key = ""

def emotion(your_day):
    # Generate text using the GPT model
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        n=1,
        messages=[
            {"role": "system", "content": "내가 오늘 하루 어땠는지에 대한 대답을 보낼거야. 나의 대답의 감정을 긍정, 부정, 중립 중 하나를 선택해서 말해줘. 단어만 보내줘. 감정이 안드러나면 중립이라고 해. 특별한 감정 없이 무슨일을 했다고 하면 중립이라고 해." + your_day},
            {"role": "user", "content": "내가 오늘 하루 어땠는지에 대한 대답을 보낼거야. 나의 대답의 감정을 긍정, 부정, 중립 중 하나를 선택해서 말해줘. 단어만 보내줘. 감정이 안드러나면 중립이라고 해. 특별한 감정 없이 무슨일을 했다고 하면 중립이라고 해." + your_day},
        ])

    message = response.choices[0]['message']
    print("{}: {}".format(message['role'], message['content']))
    print(message['content'])
    return message['content']

def chatting(your_day):
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        n=1,
        messages=[
            {"role": "system", "content": "나랑 대화하자, 너랑 나랑은 처음만났고 친해지고 싶은 관계야. 대화를 할때 주로 질문 위주로 해줘. 대답은 한 문장만 해. 공감을 잘해줘. 사람이 말하는것처럼 말해줘. 반말로 말해줘. 친근하게 말해줘. 내가 말한 것에 상관 없는 말을 하지 말아줘.\n" + your_day},
            {"role": "user", "content": "나랑 대화하자, 너랑 나랑은 처음만났고 친해지고 싶은 관계야. 대화를 할때 주로 질문 위주로 해줘. 대답은 한 문장만 해. 공감을 잘해줘. 사람이 말하는것처럼 말해줘. 반말로 말해줘. 친근하게 말해줘. 내가 말한 것에 상관 없는 말을 하지 말아줘.\n" + your_day},
        ])
    
    message = response.choices[0]['message']
    print(message['content'])

    userSay = ""

    while "대화 종료" not in userSay:

        text_to_speech(message['content'])
        oled.o_get()
        userSay = stt()
        oled.o_agree()

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            n=1,
            messages=[
                {"role": "system", "content": "너랑 나랑은 처음만났고 친해지고 싶은 관계야. 대화를 할때 주로 질문 위주로 해줘. 대답은 한 문장만 해. 공감을 잘해줘. 사람이 말하는것처럼 말해줘. 반말로 말해줘. 친근하게 말해줘. 내가 말한 것에 상관 없는 말을 하지 말아줘.\n" + userSay},
                {"role": "user", "content": f"{userSay}"},
            ])
        
        message = response.choices[0]['message']
        print(message['content'])
    
    text_to_speech("그럼 이제 너와 친해지기 위해 여러가지 활동을 준비해 보았어.")
