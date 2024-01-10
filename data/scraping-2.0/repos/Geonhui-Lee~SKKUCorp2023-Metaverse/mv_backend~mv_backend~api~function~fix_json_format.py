# Description: GetQuiz.cs(Unity)와 연결된 파일, 백엔드(quiz_generator.py)에서 생성한 퀴즈 값이 에러가 뜬다면, 이 파일을 통해 다시 수정하게 함 #
import json
from django.http import JsonResponse
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from mv_backend.lib.common import CommonChatOpenAI
from mv_backend.settings import OPENAI_API_KEY
import openai

openai.api_key = OPENAI_API_KEY
chat = CommonChatOpenAI()


# JSON 형식에 문제가 있으면, 수정 요청 또한, 예시 형식에 맞춰서 수정하라고 알려줌
npc_template = """
{input}

If this json input has a wrong json format, fix the json format and make it a correct json format.
Also, make sure that the input is like the example below, especially the explanation part.
{example}
"""

example = """{
    "quiz1": {
        "question": "다음 중 올바른 문장은 무엇인가요?",
        "choices": ["We is playing games.", "They are singing.", "He am reading a book.", "She liking to dance."],
        "answer": "They are singing.",
        "explanation": "1. <color=red>We is playing games.</color> -> <color=green>We are playing games</color>: 주어 'We'에 맞는 동사 'is' 대신 'are'를 사용해야 합니다.2. <color=green>They are singing.</color>: 주어 'They'에 맞는 동사 'are'가 사용되었습니다.3. <color=red>He am reading a book.</color> -> <color=green>He is reading a book.</color>:주어 'He'에 맞는 동사 'am' 대신 'is'를 사용해야 합니다.4. <color=red>She liking to dance.</color> -> <color=green>She likes to dance.</color>:'She'와 'liking'이 함께 사용될 때는 동사의 기본형이 사용되어야 합니다. "
    },
    "quiz2": {...},
    "quiz3": {...}
}"""

npc_prompt = PromptTemplate(
    input_variables= ["input","example"],
    template=npc_template
)

# 요청이 들어오면, JSON 형식을 확인하고, 문제가 있으면 수정하라고 알려줌
def call(request):

    npc_llm = LLMChain(
    llm=chat,
    prompt=npc_prompt
    )
    print(request.body.decode('utf-8'))
    npc_response = npc_llm.run(input = request.body.decode('utf-8'), example = example)
    print()
    print(npc_response)

    # 수정한 값을 밑의 json 형식에 맞춰서 다시 보냄
    return JsonResponse({
        "npc_response" : npc_response
    })