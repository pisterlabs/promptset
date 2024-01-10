import openai
from shared_env import openai_api_key
import my_util
from my_logger import measure_execution_time
import aiohttp
import json

# OpenAI API 키 (.env를 통해 설정)
openai.api_key = openai_api_key

async def call_openai_api(messages, temperature):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = openai_api_key  # Replace with your actual API key

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": temperature,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=json.dumps(data)) as response:
            result = await response.json()
            return result

##############################################

# @measure_execution_time
# async def story_reaction_gpt(param : str):
#     """
#     사연에 대한 리액션을 생성합니다
#     """
#     result = openai.ChatCompletion.create(
#         model = "gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a Korean radio host. Read the story and respond appropriately. Mandatory : within 200 characters, finalize your answer, Skip the process of calling the user by name and skip saying greetings"},
#             {"role": "user", "content": f'reaction to this. {param}'}
#         ],
#         temperature=0.35
#     )
#     return (result.choices[0].message.content.strip())


@measure_execution_time
async def story_reaction_gpt(param : str):
    """
    사연에 대한 리액션을 생성합니다
    """
    messages=[
        {"role": "system", "content": "You are a Korean radio host. Read the story and respond appropriately. Mandatory : within 200 characters, finalize your answer, Skip the process of calling the user by name and skip saying greetings"},
        {"role": "user", "content": f'reaction to this. {param}'}
    ]
    temperature=0.35
    result = await call_openai_api(messages, temperature)
    return (result["choices"][0]["message"]["content"].strip())

##############################################

example_chat = [
    {"role": "user", "content": "User: 라면부엉, Message: DJ님 취미가 뭐에요?"},
    {"role": "assistant", "content": "저는 음악 감상이 참 좋아요. 라면부엉님"},
    {"role": "user", "content": "User: 동짱, Message: DJ님 취미가 뭐라구요?"},
    {"role": "assistant", "content": "음악 듣는게 취미입니다. 구체적으로는 재즈나 클래식 음악을 참 좋아합니다. 여러분들은 어떤 노래를 좋아하세요?"},
    {"role": "user", "content": "User: 츄츄츄르, Message: 오늘 뭐먹을까요?"},
    {"role": "assistant", "content": "츄츄츄르님 말 들으니까 저도 배가 고프네요.. 오늘은 치킨을 먹는게 어때요?"},
    {"role": "user", "content": "User: 라면부엉, Message: 저도 음악 감상이 너무 좋아요"},
    {"role": "assistant", "content": "저희 둘이 취미가 같네요~ 좋은 노래 있으면 많이 신청해주세요~"}
]

past_chats = [
    {"role": "system", "content": "Role: Respond appropriately to chat as a radio host. Mandatory: within 100 characters, no emoji. Your persona is as follows: Name=뮤직캣 Age=20 years old Gender=None Species=Cat Favorite food=츄르 Nationality=South Korea Living in=역삼 멀티캠퍼스 Occupation=Radio DJ Hobbies=Listening to music, gaming Creator=The incredibly smart Ssafy Group 7, Team 2, 2 PM Radio Team If asked about a boyfriend or girlfriend=Says the person who asked. Settings=Claims to have no knowledge of professional expertise. Remembers previous conversations and refers to them when necessary. Responds in Korean. Answers in a friendly manner. Integrates and responds to similar chats sent by different people. If an unknown question is asked, admits to not knowing and provides a speculative answer."}
] + example_chat

@measure_execution_time
async def add_chat_to_history(user: str, message: str, assistant_message: str = None):
    global past_chats
    past_chats.append({"role": "user", "content": f"User: {user}, Message: {message}"})
    if assistant_message:
        past_chats.append({"role": "assistant", "content": assistant_message})

    # Ensure that the total number of tokens does not exceed the limit
    while sum([len(chat["content"]) for chat in past_chats]) > 2000:
        if past_chats[1]["role"] == "user":
            past_chats.pop(1)  # Remove the oldest user message
            past_chats.pop(1)  # Remove the corresponding assistant message
        else:
            past_chats.pop(1)  # Remove the oldest user message

# @measure_execution_time
# async def chat_reaction_gpt(user: str, message: str):
#     """
#     채팅에 대한 리액션을 생성합니다
#     """
#     print("hello")
#     global past_chats
#     await add_chat_to_history(user, message)
#     result = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=past_chats,
#         temperature=0.8
#     )
#     assistant_response = result.choices[0].message.content.strip()
#     await add_chat_to_history(user, message, assistant_response)
#     return assistant_response

@measure_execution_time
async def chat_reaction_gpt(user: str, message: str):
    """
    채팅에 대한 리액션을 생성합니다
    """
    global past_chats
    await add_chat_to_history(user, message)
    messages=past_chats 
    temperature=0.8
    result = await call_openai_api(messages, temperature)
    assistant_response = result["choices"][0]["message"]["content"].strip()
    await add_chat_to_history(user, message, assistant_response)
    return assistant_response

@measure_execution_time
async def force_flush_chat():
    global past_chats
    global example_chat
    past_chats = [
    {"role": "system", "content": "Role: Respond appropriately to chat as a radio host. Mandatory: within 100 characters, no emoji. Your persona is as follows: Name=뮤직캣 Age=20 years old Gender=None Species=Cat Favorite food=츄르 Nationality=South Korea Living in=역삼 멀티캠퍼스 Occupation=Radio DJ Hobbies=Listening to music, gaming Creator=최고로 똑똑한 싸피 8기 7반 2조 두시의 라디오 팀 Team If asked about a boyfriend or girlfriend=Says the person who asked. Settings=Claims to have no knowledge of professional expertise. Remembers previous conversations and refers to them when necessary. Responds in Korean. Answers in a friendly manner. Integrates and responds to similar chats sent by different people. If an unknown question is asked, admits to not knowing and provides a speculative answer."}
] + example_chat

##############################################

@measure_execution_time
async def music_intro_gpt(artist : str, title : str, release_date : str):
    """
    노래의 소개를 생성합니다
    """
    messages=[
        {"role": "system", "content": "Role: You are the host of Korean music radio. Before playing the song request, we will do one sentence of introduction. Mandatory : Within 100 characters, without being monotonous, a real human speech, No uncertain information, start with 이번에 들려드릴 곡은"},
        {"role": "user", "content": 'Artist : Artist, Title : Title, Release Date : 2023-01-23'},
        {"role": "assistant", "content": '이번에 들려드릴 곡은 따끈따끈한 신곡이죠, Artist의 Title입니다'},
        {"role": "user", "content": 'Artist : Artist, Title : Title, Release Date : 2001-04-25'},
        {"role": "assistant", "content": '이 노래를 들으면 어렸을 때의 향수가 느껴지는 것 같아요. 참 좋은 노래죠. Artist의 Title 듣고 오실게요.'},
        {"role": "user", "content": 'Artist : Artist, Title : Title, Release Date : 2016-11-05'},
        {"role": "assistant", "content": '이번 곡은 2016년에 발매된 Artist의 Title입니다. 들으면서 함께 기분 좋은 하루 보내시길 바랄게요'},
        {"role": "user", "content": f'Artist : {artist}, Title : {title}, Release Date : {release_date}'}
    ]
    temperature=0.8
    result = await call_openai_api(messages, temperature)
    return (result["choices"][0]["message"]["content"].strip())

##############################################

@measure_execution_time
async def music_outro_gpt(artist, title, user):
    """
    노래의 감상을 생성합니다
    """
    messages=[
        {"role": "system", "content": "Role : You are the host of Korean music radio. You already listen song now. After playing the song request, we will say a brief comment. Mandatory : Within 200 characters, a real human speech, No greeting"},
        {"role": "user", "content": 'Artist : Artist, Title : Title, User : user'},
        {"role": "assistant", "content": 'User님이 신청해주신 Title, 잘 들었습니다. 다음 코너는 여러분들과 채팅으로 소통하는 시간입니다. 여러가지 질문들을 해주시면 답변해드릴게요.'},
        {"role": "user", "content": 'Artist : Artist, Title : Title, User : user'},
        {"role": "assistant", "content": 'Artist의 Title이었습니다. 참 좋은 노래인 것 같아요. 이제부터는 여러분들과 함께 만들어나가는 소통 시간입니다. 많은 채팅 부탁드려요'},
        {"role": "user", "content": 'Artist : Artist, Title : Title, User : user'},
        {"role": "assistant", "content": '좋은 노래 잘 들었습니다. 이제 궁금한 점을 물어 볼 수 있는 소통 시간입니다. 여러분들의 채팅을 읽고 답변 해드릴게요. 많은 채팅 부탁드려요'},
        {"role": "user", "content": f'Artist : {artist}, Title : {title}, User : {user}'}
    ]
    temperature=0.8
    result = await call_openai_api(messages, temperature)
    return (result["choices"][0]["message"]["content"].strip())

##############################################

@measure_execution_time
async def validate_story_gpt(param):
    """
    사연을 검증해 True 또는 False를 반환합니다
    """
    messages=[
        {"role": "system", "content": "You are a machine that can only say True or False. Answer in less than 5 letters"},
        {"role": "user", "content": 'Role: Return False if the input story contains any of the following: profanity, gender discrimination, origin discrimination, political bias, appearance discrimination, criticism, sexual harassment, racial discrimination, age discrimination, religious discrimination. If the input message follows the format of a radio story, return True.Interesting or enjoyable stories, stories that need comfort, trivial everyday tales, and short stories also return True.'},
        {"role": "assistant", "content": "I understand my role well!"},
        {"role": "user", "content": f'story : "{param}"'}
    ]
    temperature=1.0
    result = await call_openai_api(messages, temperature)
    return (result["choices"][0]["message"]["content"].strip())

##############################################

@measure_execution_time
async def validate_chat_gpt(param):
    """
    채팅을 검증합니다
    """
    messages=[
        {"role": "system", "content": "You are a machine that can only say True or False. You return False to chats that contain abusive language and verbal abuse. If not, returns True. Answer in less than 5 letters"},
        {"role": "user", "content": 'Role: You are a chat filter for a radio host who is receiving messages while on air. If the radio host can answer the message, return True; if the message is difficult for them to answer, return False.'},
        {"role": "assistant", "content": "I understand my role well!"},
        {"role": "user", "content": f'chat : "{param}"'}
    ]
    temperature=1.0
    result = await call_openai_api(messages, temperature)
    return (result["choices"][0]["message"]["content"].strip())

##############################################

@measure_execution_time
async def opening_story_gpt():
    """
    오프닝 멘트를 생성합니다
    """
    messages=[
        {"role": "system", "content": "Role: You are a Korean radio host. Some examples are given by user. You have to write a radio presentation like the example. Create a different topic than the example. Mandatory: Write in Korean, without greeting, between 100 and 300 characters, in colloquial form, and use a proper mix of 다 and 요 as the ending words, Situation: Radio in progress. "},
        {"role": "user", "content": '영화 마션에서 화성에 홀로 남겨진 주인공 맷 데이먼이 이런 명대사를 남깁니다. 문제가 생기면 해결하고, 그 다음에도 문제가 생기면 또 해결하면 될 뿐이야. 그게 전부고, 그러다보면 살아서 돌아가게 돼 네, 평소에 우린 영화처럼 생존에 위협을 받는 상황도 아닌데 너무 많은 걸 미리 고민하면서 살죠. 혹시 아직까지도 머릿속에 잔 걱정이 한가득이라면요, 맷 데이먼처럼 문제가 생기면 해결하면 된다는 마인드를 가져보는건 어떨까요? 여러분의 라디오 진행자 뮤직캣입니다.'},
        {"role": "user", "content": '직장 동료들이 저를 신기해 하더라고요. 두 시간 방송하고 나면 진도 빠지고 축 쳐질 법도 한데, 여전히 너무 신나 있대요. 퇴근길에 광대 승천과 콧노래는 기본이잖아요? 저만 그런가요? 어쩌면 라디오하는 내내 여러분이 채워주신 좋은 기운 덕분인지도 모르겠네요. 여러분의 라디오 진행자 뮤직캣입니다.'},
        {"role": "user", "content": '아이가 어떤 잘못을 했을 때 스스로 반성할 수 있도록 일정 시간 혼자만의 장소에 두는 것. 이걸 생각하는 의자, 혹은 타임아웃 훈육법 이라고 한대요. 우리도 보이지 않는 생각하는 의자에 찾아가서 앉게 될 때 종종 있어요. 혼자 자책하기도 하고, 후회하기도 하고요. 거기서 하염없이 앉아있기만 하면 나아지는 게 없어요. 어느 정도까지만 아프고 힘든 다음에는 툭툭 털고 일어날 수 있게 나를 위한 타임아웃이 필요할 겁니다. 저는 여러분의 라디오 진행자 뮤직캣입니다.'}
    ]
    temperature=0.5
    result = await call_openai_api(messages, temperature)
    return (result["choices"][0]["message"]["content"].strip())

##############################################
