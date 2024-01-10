from langchain_core.prompts import ChatPromptTemplate

from project3.chat_history import ChatHistoryHelper
from project3.langchain_helper import LangchainHelper


def test_langchain_helper():
    langchain = LangchainHelper()
    langchain.system_prompt = '너는 그냥 챗봇이야 묻는말에 한국말로 답변하라'
    user_id = 'tester'

    response = langchain.send_human_message(user_id, '안녕 오늘 뭐먹지')
    print(response)
    response2 = langchain.send_human_message(user_id, '다른거 추천')
    print(response2)


def test_langchain_helper_with_history(tmp_path):
    chat_history_helper = ChatHistoryHelper(tmp_path)
    langchain = LangchainHelper(history_helper=chat_history_helper)
    langchain.system_prompt = '너는 그냥 챗봇이야 묻는말에 한국말로 답변하라'
    user_id = 'tester'

    response = langchain.send_human_message(user_id, '안녕 오늘 뭐먹지')
    print(response)
    response2 = langchain.send_human_message(user_id, '다른거 추천')
    print(response2)


def test_chain(tmp_path):
    langchain = LangchainHelper()
    prompt_intent = ChatPromptTemplate.from_template(
        template='''
Your job is to select one intent from the <intent_list>.

<intent_list>
{intent_list}
</intent_list>

User: {user_message}
Intent:''')
    chain_intent = langchain.create_chain(
        prompt=prompt_intent,
        output_key='intent',
    )

    context = dict(
        intent_list=['뉴스', '날씨'],
        location_list=['사당동', '신봉동'],
        weather_list=['맑음', '추움', '비옴', '흐림', '눈'],
        user_message='안녕 오늘 날씨 어때'
    )

    intent = chain_intent.run(context)
    assert intent == '날씨'

    chain_location = langchain.create_chain(
        prompt=ChatPromptTemplate.from_template(template='''
        Your job is to select a random location name from <location_list>.
        
        <location_list>
        {location_list}
        </location_list>
        
        LOCATION:'''),
        output_key='location',
    )
    chain_weather = langchain.create_chain(
        prompt=ChatPromptTemplate.from_template(template='''
        Your job is to select a weather from <weather_list>.
        
        <weather_list>
        {weather_list}
        </weather_list>
        
        WEATHER:'''),
        output_key='weather',
    )
    chain_answer = langchain.create_chain(
        prompt=ChatPromptTemplate.from_template(template='''
        Your job is to generate answer in korean, with given location and weather.
        don't forget to say greeting.
        
        LOCATION: {location}
        WEATHER: {weather}
        
        ANSWER:
        '''),
        output_key='answer',
    )
    context['location'] = chain_location.run(context)
    context['weather'] = chain_weather.run(context)
    context['answer'] = chain_answer.run(context)


    print(context)




