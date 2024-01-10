import asyncio
import traceback
import openai
from sanic import Request, response
from genaipf.conf.server import os, IS_INNER_DEBUG
from genaipf.utils.log_utils import logger
from genaipf.constant.error_code import ERROR_CODE
from genaipf.interfaces.common_response import success, fail
from genaipf.dispatcher.api import aref_oneshot_gpt_generator
async def send_oneshot_chat(request: Request):
    '''
    messages: [
        {'role': 'user_a', 'content': 'how are you?', 'type': 'text', 'format': 'text', 'version': 'v001'},
        {'role': 'user_b', 'content': '不错', 'type': 'text', 'format': 'text', 'version': 'v001'}
    ]
    '''
    try:
        logger.info("======start send_oneshot_chat===========")
        request_params = request.json
        messages = request_params.get('messages', [])
        model = request_params.get('model', 'ml-plus')
        language = request_params.get('language', 'en')
        preset_name = request_params.get('preset_name', None)
        target_language = request_params.get('target_language', 'English')
        token = request_params.get('token', '')
        
        data = {
            "messages": messages,
            "target_language": target_language,
            "token": token
        }
        
        resp = await aref_oneshot_gpt_generator(
            messages=messages,
            model=model,
            language=language,
            preset_name=preset_name,
            picked_content=[],
            related_qa=[],
            data=data
        )
        _tmp_text = resp.choices[0].message.content
        resp_msgs = {
            "messages": [
                {'role': messages[-1]['role'], 'content': _tmp_text, 'type': 'text', 'format': 'text', 'version': 'v001'}
            ]
        }
        return success(resp_msgs)
    except Exception as e:
        logger.error(str(e))
        return fail(ERROR_CODE['PARAMS_ERROR'], str(e))
    