import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import openai
from sanic import Request, response
from sanic.response import ResponseStream
from genaipf.exception.customer_exception import CustomerError
from genaipf.constant.error_code import ERROR_CODE
from genaipf.interfaces.common_response import success,fail
import requests
import json
# import snowflake.client
import genaipf.services.gpt_service as gpt_service
from genaipf.controller.preset_entry import preset_entry_mapping
import genaipf.services.user_account_service_wrapper as user_account_service_wrapper
from datetime import datetime
from genaipf.utils.log_utils import logger
import time
from pprint import pprint
from genaipf.dispatcher.api import gpt_functions, afunc_gpt4_generator, aref_answer_gpt_generator
from genaipf.dispatcher.utils import get_qa_vdb_topk, merge_ref_and_input_text
from genaipf.dispatcher.prompts_v001 import LionPrompt
# from dispatcher.gptfunction import unfiltered_gpt_functions, gpt_function_filter
from genaipf.dispatcher.functions import gpt_functions_mapping, gpt_function_filter
from genaipf.dispatcher.postprocess import posttext_mapping, PostTextParam
from genaipf.utils.redis_utils import RedisConnectionPool
from genaipf.conf.server import IS_INNER_DEBUG, IS_UNLIMIT_USAGE
from genaipf.utils.speech_utils import transcribe, textToSpeech
import os
import base64
from dotenv import load_dotenv
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

proxy = { 'https' : '127.0.0.1:8001'}

executor = ThreadPoolExecutor(max_workers=10)

async def http(request: Request):
    return response.json({"http": "sendchat"})


async def http4gpt4(request: Request):
    return response.json({"http4gpt4": "sendchat_gpt4"})

def process_messages(messages):
    processed_messages = []
    for message in messages:
        if message.get('type') == 'voice':
            content = transcribe(message['content'])
            need_whisper = True
        else:
            content = message['content']
            need_whisper = False
        processed_messages.append({
            "role": message['role'],
            "content": content,
            "type": message.get('type', 'text'),
            "format": message.get('format', 'text'),
            "version": message.get('version', 'v001'),
            "need_whisper": need_whisper
        })
    return processed_messages[-10:]

async def send_strem_chat(request: Request):
    logger.info("======start gptstream===========")

    request_params = request.json
    # if not request_params or not request_params['content'] or not request_params['msggroup']:
    #     raise CustomerError(status_code=ERROR_CODE['PARAMS_ERROR'])

    userid = 0
    if hasattr(request.ctx, 'user'):
        userid = request.ctx.user['id']
    language = request_params.get('language', 'en')
    msggroup = request_params.get('msggroup')
    messages = request_params.get('messages', [])
    device_no = request.remote_addr
    question_code = request_params.get('code', '')
    model = request_params.get('model', '')
    # messages = process_messages(messages)
    output_type = request_params.get('output_type', 'text') # text or voice; (voice is mp3)
    # messages = [{"role": msg["role"], "content": msg["content"]} for msg in process_messages(messages)]
    # messages = messages[-10:]
    messages = process_messages(messages)
    try:
        if (not IS_UNLIMIT_USAGE and not IS_INNER_DEBUG) and model == 'ml-plus':
            can_use = await user_account_service_wrapper.get_user_can_use_time(userid)
            if can_use > 0:
                await user_account_service_wrapper.minus_one_user_can_use_time(userid)
            else:
                raise CustomerError(status_code=ERROR_CODE['NO_REMAINING_TIMES'])
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())

    try:
        async def event_generator(_response):
            # async for _str in getAnswerAndCallGpt(request_params['content'], userid, msggroup, language, messages):
            async for _str in getAnswerAndCallGpt(request_params.get('content'), userid, msggroup, language, messages, device_no, question_code, model, output_type):
                await _response.write(f"data:{_str}\n\n")
                await asyncio.sleep(0.01)
        return ResponseStream(event_generator, headers={"accept": "application/json"}, content_type="text/event-stream")

    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())


   

async def  getAnswerAndCallGpt(question, userid, msggroup, language, front_messages, device_no, question_code, model, output_type):
    t0 = time.time()
    MAX_CH_LENGTH = 8000
    _ensure_ascii = False
    messages = []
    for x in front_messages:
        if x.get("code"):
            del x["code"]
        if x["role"] == "gptfunc":
            messages.append({"role": "assistant", "content": None, "function_call": x["function_call"]})
        else:
            messages.append({"role": x["role"], "content": x["content"]})
    user_history_l = [x["content"] for x in messages if x["role"] == "user"]
    newest_question = user_history_l[-1]
    data = {}
    
    last_front_msg = front_messages[-1]
    question = last_front_msg['content']
    if last_front_msg.get("need_whisper"):
        yield '[WHISPER]'
        yield json.dumps({"text": last_front_msg['content']})
    
    # vvvvvvvv 在第一次 func gpt 就准备好数据 vvvvvvvv
    ref_text = ""
    related_qa = get_qa_vdb_topk(newest_question)
    ref_text += "\n\n可能相关的历史问答:\n" + "\n\n".join(related_qa)
    ref_text = ref_text[:MAX_CH_LENGTH + 3000]
    logger.info(f'>>>>> frist ref_text: {ref_text}')
    merged_ref_text = merge_ref_and_input_text(ref_text, newest_question, language=language)
    _messages = [x for x in messages if x["role"] != "system"]
    # msgs = _messages[:-1] + [{"role": "user", "content": merged_ref_text}]
    msgs = _messages[::]
    # ^^^^^^^^ 在第一次 func gpt 就准备好数据 ^^^^^^^^

    used_gpt_functions = gpt_function_filter(gpt_functions_mapping, _messages)
    # resp1 = await afunc_gpt4_generator(msgs, used_gpt_functions, language, model)
    resp1 = await afunc_gpt4_generator(msgs, used_gpt_functions, language, model, "", related_qa)
    chunk = await resp1.__anext__()
    # _func_or_text = chunk['choices'][0]['delta'].get("function_call", None)
    _func_or_text = chunk.choices[0].delta.function_call
    if _func_or_text:
        mode1 = "func"
        logger.info(f">>>>> activate gpt function >>>>>")
    else:
        mode1 = "text"
    if mode1 == "text":
        # c0 = chunk['choices'][0]['delta'].get("content", "")
        c0 = chunk.choices[0].delta.content
        _tmp_text = ""
        _tmp_text += c0
        yield '[GPT]'
        _code = generate_unique_id()
        yield json.dumps({"code": _code})
        yield json.dumps({"text": c0})
        async for chunk in resp1:
            # _gpt_letter = chunk['choices'][0]['delta'].get("content", "")
            _gpt_letter = chunk.choices[0].delta.content
            if _gpt_letter:
                _tmp_text += _gpt_letter
                yield json.dumps({"text": _gpt_letter})
        if output_type == "voice":
            # 对于语音输出，将文本转换为语音并编码
            base64_encoded_voice = textToSpeech(_tmp_text)
            yield '[TTS]'
            yield json.dumps({
                "role": "assistant", 
                "type": "voice", 
                "format": "mp3", 
                "version": "v001", 
                "content": base64_encoded_voice
            })
        yield "[DONE]"
        data = {
                'type' : 'gpt',
                'content' : _tmp_text,
                'code' : _code
            }
        logger.info(f'>>>>> text _tmp_text: {_tmp_text}')
    elif mode1 == "func":
        big_func_name = _func_or_text.name
        func_name, sub_func_name = big_func_name.split("_____")
        _arguments = _func_or_text.arguments
        async for chunk in resp1:
            # _func_json = chunk['choices'][0]['delta'].get("function_call", {})
            _func_json = chunk.choices[0].delta.function_call
            # print(f'>>>>> _func_json: {_func_json}')
            if _func_json:
                _arguments += _func_json.arguments
            # _arguments += _func_json.get("arguments", "")
        _param = json.loads(_arguments)
        _param["language"] = language
        _param["subtype"] = sub_func_name
        logger.info(f'>>>>> func_name: {func_name}, sub_func_name: {sub_func_name}, _arguments: {_arguments}')
        t01 = time.time()
        logger.info(f'>>>>> gpt func time: {t01 - t0}')
        content = ""
        _type = ""
        presetContent = {}
        picked_content=""
        if func_name in preset_entry_mapping:
            preset_conf = preset_entry_mapping[func_name]
            _type = preset_conf["type"]
            _args = [_param.get(x) for x in preset_conf["param_names"]]
            presetContent, picked_content = await preset_conf["get_and_pick"](*_args)
            if preset_conf.get("has_preset_content") and (_param.get("need_chart") or preset_conf.get("need_preset")):
                data = {
                    'type' : _type,
                    'subtype': sub_func_name,
                    'content' : content,
                    'presetContent' : presetContent
                }

        related_qa = get_qa_vdb_topk(newest_question)
        # merged_ref_text = LionPrompt.get_merge_ref_and_input_prompt(str(picked_content), related_qa, newest_question, language, _type, data)
        # merged_ref_text = merge_ref_and_input_text(ref_text, newest_question)
        _messages = [x for x in messages if x["role"] != "system"]
        # msgs = _messages[:-1] + [{"role": "user", "content": merged_ref_text}]
        msgs = _messages[::]
        # resp2 = await aref_answer_gpt_generator(msgs, model="gpt-3.5-turbo-16k", language=language, preset_name=_type)
        resp2 = await aref_answer_gpt_generator(msgs, model, language, _type, str(picked_content), related_qa)

        # if data :
        #     yield '[DATA]'
        #     yield json.dumps(data)
        _tmp_text = ""
        t1 = time.time()
        logger.info(f'>>>>> get data time: {t1 - t01}')
        logger.info(f'>>>>> start->data done t1 - t0: {t1 - t0}')
        # if func_name in preset_entry_mapping:
        #     yield '[GPTFUNC]'
        #     _gptfunc_data = {"role": "gptfunc", "function_call": {"name": big_func_name, "arguments": _arguments}}
        #     yield json.dumps(_gptfunc_data)
        yield "[GPT]"
        async for chunk in resp2:
            # _gpt_letter = chunk['choices'][0]['delta'].get("content", "")
            _gpt_letter = chunk.choices[0].delta.content
            # print(f'>>>>> _gpt_letter: {_gpt_letter}')
            if _gpt_letter:
                _tmp_text += _gpt_letter
                yield json.dumps({"text": _gpt_letter})
        posttexter = posttext_mapping.get(func_name)
        if posttexter is not None:
            async for _gpt_letter in posttexter.get_text_agenerator(PostTextParam(language, sub_func_name)):
                _tmp_text += _gpt_letter
                yield json.dumps({"text": _gpt_letter})
        if output_type == "voice":
            # 对于语音输出，将文本转换为语音并编码
            base64_encoded_voice = textToSpeech(_tmp_text)
            yield '[TTS]'
            yield json.dumps({
                "role": "assistant", 
                "type": "voice", 
                "format": "mp3", 
                "version": "v001", 
                "content": base64_encoded_voice
            })
        if len(data) == 0 :
            data = {
                'type' : 'gpt',
                'content' : _tmp_text
            }
        else :
            data['content'] = _tmp_text
        # print(f'>>>>>test 002 : {data}')
        if data:
            _code = generate_unique_id()
            data['code'] = _code
            yield '[DATA]'
            yield json.dumps(data)
        yield "[DONE]"
        logger.info(f'>>>>> func & ref _tmp_text & output_type: {output_type}: {_tmp_text}')

    if question and msggroup :
        gpt_message = (
        question,
        'user',
        userid,
        msggroup,
        question_code,
        device_no
        )
        await gpt_service.add_gpt_message_with_code(gpt_message)
        if data['type'] == 'coin_swap':  # 如果是兑换类型，存库时候需要加一个过期字段，前端用于判断不再发起交易
            data['expired'] = True
        messageContent = json.dumps(data)
        gpt_message = (
            messageContent,
            data['type'],
            userid,
            msggroup,
            data['code'],
            device_no
        )
        await gpt_service.add_gpt_message_with_code(gpt_message)

    


def generate_unique_id():
    redis_client = RedisConnectionPool().get_connection()
    return redis_client.incr('unique_id')