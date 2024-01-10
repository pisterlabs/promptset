# Description: GetQuiz.cs(Unity)와 연결된 파일, 퀴즈를 정해진 형식으로 만드는 chatgpt #
import json
from django.http import JsonResponse
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from mv_backend.settings import OPENAI_API_KEY
from mv_backend.lib.common import CommonChatOpenAI
from mv_backend.lib.database import Database
import openai

db = Database()
openai.api_key = OPENAI_API_KEY
chat = CommonChatOpenAI()

#quiz prompt
# 사용자의 미진 사항(retireve), 사용자의 특징(reflect), 사용자의 CEFR 수준을 받아서, 그에 맞는 퀴즈를 만들어줌

# 3개의 퀴즈를 만들어야 하고, 퀴즈는 문제, 4개의 선택지, 정답, 해설로 이루어져 있음 (문제와 해설은 한국어, 선택지와 정답은 영어)

# 퀴즈의 문제와 해설은 사용자의 미진 사항과 특징을 참고해서 만들어야 함, 퀴즈의 선택지와 정답은 사용자의 CEFR 수준을 참고해서 만들어야 함

# 해설은 틀린 선택지의 올바른 문장을 적어야 함

npc_template = """
user is bad at:
{retrieve}

user's characteristics:
{reflect}

CEFR is the English-level criteria established by the Common European Framework of Reference for Languages, which ranges from A1 to C2 (pre-A1, A1, A2, B1, B2, C1, C2).
User's CEFR level: "{user_cefr}"

You are a quiz maker for the User. You have to make 3 quizzes(only make 3 quiz) that matches the CEFR level of the user that helps solve the bad parts user has.
If the user is bad at grammer, make a quiz that helps user to learn grammer by referring "user is bad at" and "reason: ".(*Make sure to make a quiz that is based on the topic of the user's grammer mistakes.*)
If the user has a problem with his/hers attitude, make a quiz that helps user to be polite by referring "user is bad at" and "reason: ".
Also, make one quiz asking a question related to the topic of conversation in user's characteristics.
The quiz should have a question, 4 choices, a answer, and a explanation. Question, and the explenation must be written in **KOREAN(한국어로)**(always Korean), And the choices and the answer must be written in *English*.
(The Explanation must include *specific reason* why the other choices are wrong(*Always* Include *the corrected full sentence* of the incorrect answer.)and why the answer is correct each **KOREAN(한국어로)**(always Korean). )-> the format of the explanation is in the example below.
(The choices must have ***ONLY ONE*** correct answer, and *three* wrong answers).

Make 3 quiz and write it down in a below example format.

The order of correct and wrong choices in the example is *not fixed* (*Must* make it in random order). Make quizs that fits "user is bad at" and "reason: ". *Must* include "<color=red>" and "</color>".
format) {example}

"""

#예시 형식
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
    input_variables= ["retrieve","reflect","user_cefr","example"],
    template=npc_template
)

def call(request):
    body_unicode = request.body.decode('utf-8') 
    body = json.loads(body_unicode)

    user_name = body['username']

    #사용자의 미진 사항 DB에서 가져오기 (가장 최근의 미진 사항 3개)
    retrieve_collection = db.get_collection(user_name, 'Retrieves')
    last_retrieves = list(retrieve_collection.find(sort=[('node', -1)], limit=3))
    retrieve_str = '\n'.join([r['retrieve'] for r in last_retrieves[::-1]])
    print(retrieve_str)

    #사용자의 특징 DB에서 가져오기 (가장 최근의 특징 3개)
    reflect_collection = db.get_collection(user_name, 'Reflects')
    last_reflects = list(reflect_collection.find(sort=[('node', -1)], limit=3))
    reflect_str = '\n'.join([r['reflect'] for r in last_reflects[::-1]])
    print(reflect_str)

    #사용자의 CEFR DB에서 가져오기
    cefr_collection = db.get_collection(user_name, 'CEFR_GPT')
    #가장 최근의 cefr 가져오기
    cefr = cefr_collection.find_one(sort=[('node', -1)])['cefr']
    print(cefr)
    print()

    npc_llm = LLMChain(
    llm=chat,
    prompt=npc_prompt
    )

    # GPT 실행
    npc_response = npc_llm.run(retrieve = retrieve_str,reflect = reflect_str, user_cefr = cefr ,example = example)

    print(npc_response)

    # 밑의 형식으로 Unity에게 보냄
    return JsonResponse({
        "npc_response": npc_response
    })