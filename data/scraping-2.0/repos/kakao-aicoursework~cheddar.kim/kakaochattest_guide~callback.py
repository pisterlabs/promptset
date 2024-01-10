from dto import ChatbotRequest
from samples import list_card
import requests
import time
import logging
import openai
import os
import requests

from utils.langchain import get_single_query, get_chained_query
from utils.vectordb import create_vectordb

# 환경 변수 처리 필요!
openai.api_key = os.environ["LLM_LECTURE_KEY"]
os.environ["OPENAI_API_KEY"] = os.environ["LLM_LECTURE_KEY"]
logger = logging.getLogger("Callback")

# upload_embeddings_from_dir("datas/")
vdb = create_vectordb()

# result = get_single_query("안녕하세요", SYSTEM_MSG, 
# result = get_single_query("카카오톡 프로필 어떻게 설정하니?", vdb)
# result = get_chained_query("카카오톡 프로필 어떻게 설정하니?", vdb)
# print("result", result)

def callback_handler(request: ChatbotRequest) -> dict:

    # get a single data
    query_text = request.userRequest.utterance
    # output_text = get_single_query(query_text, vdb, temperature=0.0)["answer"]
    output_text = get_chained_query(query_text, vdb, temperature=0.0)["answer"]
    print("output_text", output_text)
    
    # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": str(output_text).strip()
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    url = request.userRequest.callbackUrl

    if url:
        requests.post(url, json=payload)
    
