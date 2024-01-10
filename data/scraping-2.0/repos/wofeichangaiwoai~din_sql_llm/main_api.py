from ubix.common.api_status import APIStatus
from ubix.common.log_basic import logging
import os
import time
import uuid
from datetime import datetime
from addict import Dict
from flask import Flask, request, jsonify, Response
from langchain import OpenAI
from threading import Lock
from route.router import router_chain, llm, get_route_meta
from ubix.common.answer import get_answer
from ubix.common.conversation_manage import get_or_create_conversion, update_conversation, get_conversation_list, \
    get_conversation_by_id, save_conversation
import json

config = {
    "DEBUG": True,          # some Flask specific configs
}

app = Flask(__name__)
# tell Flask to use the above defined config
app.config.from_mapping(config)


conversion_storage = {}
lock = Lock()

route_meta = get_route_meta()

@app.route("/")
def hello():
    return "hello"


@app.route("/get_llm_api_status", methods=['GET'])
def get_llm_api_status():
    enable_llm = request.args.get('enable_llm')
    if enable_llm:
        enable_llm = enable_llm.lower() == 'true'
    else:
        enable_llm = False
    res: Dict = APIStatus().get_api_status(enable_llm=enable_llm)
    return Response(json.dumps(res,  indent=4), mimetype='application/json')


@app.route("/get_conversation_list", methods=['GET'])
def _get_conversation_list():
    user_id = request.args.get('user_id')
    conversation_list = get_conversation_list(user_id)
    return conversation_list


@app.route("/get_conversation", methods=['GET'])
def _get_conversation():
    id = request.args.get('id')
    conversation = get_conversation_by_id(id)
    return conversation


@app.route("/update_conversation", methods=['POST'])
def _update_conversation():
    request_data = Dict(request.json).data
    logging.info(request.json)
    logging.info(request.headers)
    args = {
        "_id": request_data.conversation_id,
        "user_id":  request.headers.get("User-ID", "UNKNOWN"),
        "conversation_id": request_data.conversation_id,
    }
    if "query_type" in request_data:
        args["query_type"] = request_data.query_type
    if "conversation_name" in request_data:
        args["conversation_name"] = request_data.conversation_name
    return update_conversation(args)


@app.route("/chat/v2", methods=['POST'])
def get_bot_response_dummpy():
    start = datetime.now()
    request_data = Dict(request.json).data
    question = request_data.question
    logging.info(f"question:{question}, request data:{request_data}, header:{request.headers}")
    with lock:
        route_name, answer = get_answer(question, history=None, query_type=request_data.query_type)

    sql = None
    #print(f"answer type:{type(answer)}, route_name:{route_name}, {list(answer.keys())}")
    widgets = None
    if route_name in ["query", "download"]:
        # answer = Dict(json.loads(answer))
        sql = answer.get("sql")
        answer = answer.get("answer")
        widgets = "table"

    end = datetime.now()
    duration = (end-start).total_seconds()
    logging.info(f"Try to get the conversation")
    download_type = "download"
    conversion = get_or_create_conversion(request_data, request.headers)
    logging.info(f"Try to save the conversation")
    current_qa = {"id": uuid.uuid4().hex[:24],
                  "question": question,
                  "answer": answer,
                  "related_sql": sql,
                  "query_type": request_data.query_type,
                  "route": route_name,
                  "start_time": start.strftime('%Y-%m-%d %H:%M:%S'),
                  "duration": duration,
                  "host_name": os.environ.get("HOSTNAME", "UNKNOWN"),
                  "llm_type": os.getenv("LLM_TYPE", "UNKNOWN"),
                  "widget_type": widgets,
                  "download_type": download_type,
                  }

    conversion.conversation_list.append(current_qa)
    if route_name != "error":
        save_conversation(conversion)
        logging.info("Conversation save successfully")
    qa_response = conversion
    conversion.pop("_id")
    conversion.pop("conversation_list")
    current_qa["conversation_id"] = qa_response["conversation_id"]
    qa_response["qa_data"] = current_qa
    return qa_response


@app.route("/timeout_test")
def timeout_test():
    time.sleep(150)
    return f"delay 150 seconds"




def get_route_name(question):
    stop = None if isinstance(router_chain.llm, OpenAI) else "\n"
    route_name_original = router_chain.run(input=str(question), stop=stop)
    route_name = route_name_original.strip().split(':')[-1].strip()
    route_name = route_name if route_name in route_meta else "other"
    logging.info(f"route_name:{route_name_original}=>{route_name}")
    return route_name




if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

""""
CUDA_VISIBLE_DEVICES=0 LLM_TYPE=vllm nohup python -u main_api.py >api.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 LLM_TYPE=gglm nohup python -u main_api.py >api.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 LLM_TYPE=tgi nohup python -u main_api.py >api.log 2>&1 &


curl "http://colossal-ai.data-tooling.svc.cluster.local:5000/chat/v2" \
-H  'User-ID: user_123' \
-H  'Account-ID: acct_12345' \
-H 'Content-Type: application/json' \
--data-raw '{"data":{"query_type":"query", "question":"How many records in this table", "conversation_id":"8f187c01601c" }}' \
--compressed \
--insecure | jq

curl 'https://colossal-ai.home-dev.ubix.io/chat/v2' \
-H  'User-ID: user_123' \
-H  'Account-ID: acct_12345' \
-H 'Content-Type: application/json' \
--data-raw '{"data":{"query_type":"download", "question":"list the data with cardcode C40000 in this table", "conversation_id":"8f187c01601c" }}' \
--compressed \
--insecure | jq

curl 'https://colossal-ai.home-dev.ubix.io/update_conversation' \
-H  'User-ID: user_123' \
-H  'Account-ID: acct_12345' \
-H 'Content-Type: application/json' \
--data-raw '{"data":{"conversation_name":"hello_123", "conversation_id":"8f187c01601c" }}' \
--compressed \
--insecure | jq


curl 'http://colossal-ai.data-tooling.svc.cluster.local:5000/get_conversation_list?user_id=user_123'

curl 'http://colossal-ai.data-tooling.svc.cluster.local:5000/get_llm_api_status?enable_llm=false'

curl 'http://colossal-ai.data-tooling.svc.cluster.local:5000/get_llm_api_status?enable_llm=true'

curl 'http://colossal-ai.data-tooling.svc.cluster.local:5000/get_conversation?id=8f187c01601c'

 



curl -X POST -d "question=How many records in this table" https://colossal-ai.home-dev.ubix.io/chat

curl --connect-timeout 500 --location --request POST "https://colossal-ai.home-dev.ubix.io/chat" \
    -d "question=What is the maximum total in this table?"

curl --connect-timeout 500 --location --request POST "https://colossal-ai.home-dev.ubix.io/chat" \
    -d "question=What is black body radiation?"



curl --connect-timeout 500 --location --request POST "https://colossal-ai.home-dev.ubix.io/chat" \
    -d "question=Hello, I'm Felix"


 
curl --connect-timeout 500 --location --request POST "https://colossal-ai.home-dev.ubix.io/chat" \
    -d "question=Hello, I'm Felix"



curl --connect-timeout 500 --location --request POST "https://colossal-ai.home-dev.ubix.io/chat_dummpy" \
    -d "question=hello"



curl --connect-timeout 500 --location --request POST "http://localhost:5000/chat" \
    -d "question=Hello, I'm Felix"


time curl https://colossal-ai.home-dev.ubix.io/timeout_test

time curl http://colossal-ai/timeout_test

"""
