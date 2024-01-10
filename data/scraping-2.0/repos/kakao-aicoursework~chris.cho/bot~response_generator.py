import time

#from bot.openai_chat_processor import OpenAIChatProcessor
from bot.langchain_chat_processor import LanChainChatProcessor as ChatProcessor
from bot.conversation_manager import ConversationManager
from functions import function_selector
import constants


_DEBUG_MODE=False #디버그 정보(응답시간, 함수명, 함수 호출 여부 등) 표시 유무
_DETAIL_DEBUG_MODE=False #메모리안에 들어 있는 구체적인 내용

def process_user_input(user_input, callback, DEBUG=_DEBUG_MODE, DETAIL_DEBUG=_DETAIL_DEBUG_MODE,
                       RESULT_ANSWER_COMPRESSION_MODE=True):
    # 여기서 비동기적으로 처리 로직을 수행
    # 단기 기억 메모리 설정
    message_log = conversation_manager.get_long_term_memory()

    if DEBUG:
        max_size = conversation_manager.get_max_memory_size()
        debug_message = (f"[DEBUG] <long_term_memory_queue_size (cur/max)={len(message_log)}/{max_size}>\n")
        if DETAIL_DEBUG:#get_total_content_len
            debug_message += (f"[DETAIL_DEBUG] <total_message_len={conversation_manager.get_total_content_len()}>"
                              f"[DETAIL_DEBUG] get_long_term_memory <message_log={message_log}>\n")
    else:
        pass

    message_log.append({'role': 'user', 'content': f'{user_input}'})

    if DEBUG:
        start_time = time.time()

    (response,
     is_function_call_enabled,
     function_name,
     chat_context_dict) = chat_processor.process_chat_with_function(message_log)

    if DEBUG:
        dur_time_sec = round(time.time() - start_time, 1)
        debug_message += (f"[DEBUG] <dur_time_sec={dur_time_sec}>\n"
                         f"[DEBUG] <is_function_call_enabled={is_function_call_enabled}>\n"
                         f"[DEBUG] <function_name={function_name}>\n"
                         f"\n")
        print(f"[DEBUG] debug_message={debug_message}")
    else:
        debug_message = None

    if callback is None:
        response_message = response['choices'][0]['message']
        response_content = response_message['content']
    else:
        response_content = callback(response, debug_message, DEBUG)

    if RESULT_ANSWER_COMPRESSION_MODE:
        compressed_response_content = chat_processor.compress_result_answer(chat_context_dict, response_content)
    else:
        compressed_response_content = response_content

    conversation_manager.manage_conversation(user_input, compressed_response_content)
    return response_content


def init_chat_processor_and_conversation_manager(gpt_model ="gpt-3.5-turbo-16k"):
    global chat_processor
    global conversation_manager

    (functions,
     available_functions,
     init_memory,
     str_function_concept) = _init_function_call_info(db_name=None, is_multi_function=True)
    # 채팅 프로세스초기화

    chat_processor = ChatProcessor(gpt_model,
                                   functions=functions,
                                   available_functions=available_functions,
                                   init_memory=init_memory,
                                   init_tag = str_function_concept)  # 실제 모델 이름으로 대체

    # 대화 관리자 초기화
    conversation_manager = ConversationManager(init_memory=init_memory)

    return chat_processor, conversation_manager

def _init_function_call_info(db_name, is_multi_function=False):
    str_function_concept = ""
    init_memory = []
    functions = []
    available_functions = {}
    if not is_multi_function:
        (function,
         available_function,
         global_tag,
         default_system_log_dict) = function_selector.get_function_call_info(db_name)

        functions = [function]
        available_functions.update(available_function)
        str_function_concept += f"{global_tag}"
        #init_memory = [] #페스로나 컨셉으로 대체
    else:
        role_tag = ""
        for db_name in [constants.KAKAO_SOCIAL_GUIDES,
                        constants.KAKAO_SYNC_GUIDES,
                        constants.KAKAO_CHANNEL_GUIDES]:
            (function,
             available_function,
             global_tag,
             default_system_log_dict) = function_selector.get_function_call_info(db_name)

            functions.append(function)
            available_functions.update(available_function)
            role_tag += f"{global_tag},"
            pass

        #init_memory = [] ##페스로나 컨셉으로 대체
        str_function_concept = role_tag[:-1]

    return functions,available_functions, init_memory, str_function_concept