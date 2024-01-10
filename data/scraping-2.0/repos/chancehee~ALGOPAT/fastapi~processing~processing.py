from myclass.problem import ProblemData
from langchain.chat_models import ChatOpenAI
from processing.problem.summary_info import summary_info
from processing.problem.summary_description import summary_description
from processing.usercode.summary_code_complexity import summary_code_complexity
from processing.usercode.summary_code_refactor import summary_code_refactor
from processing.translator.translate_text import translate_texts
from myparser.problem.parse_summary import parse_summary
from myparser.usercode.parse_summary_code import parse_summary_code
from asyncio import gather
from chain.json_formatter.json_formatter import json_formatter
from database.problem import    check_gpt_problem_summary_is_exist
from database.database import save_problem_origin \
                            , save_user_problem_origin \
                            , save_problem_summary \
                            , save_user_submit_solution_origin \
                            , save_gpt_solution \
                            , check_gpt_problem_summary_is_exist \
                            , get_gpt_problem_summary \
                            , update_problem_meta
from logging import getLogger
from utils.log_decorator import time_logger
from utils.shared_env import redis_url, open_ai_api_key, decrypt_key
from utils.utils import parse_lang_type
import redis_lock
from redis import Redis
import asyncio
from my_dto.common_dto import MessageDTO, UserServiceDTO
from my_exception.exception import MyCustomError 
from database.get_session import async_session
from Crypto.Cipher import AES
from base64 import b64decode


# logger 설정 
logger = getLogger()

OPEN_BRACE = '{'
CLOSE_BRACE = '}'
LOCK_TIMEOUT = 300

local_open_ai_api_key = open_ai_api_key # 서버에서 들고있는 OPEN_AI_API_KEY 

redis_conn = Redis.from_url(redis_url)  # Redis 연결 설정 정보

SECRET_KEY = decrypt_key
cipher = AES.new(SECRET_KEY.encode(), AES.MODE_ECB)
    
# GPT 응답 생성 함수
@time_logger("GPT 비즈니스 로직")
async def processing(data : ProblemData, send_callback):
    user_seq = data.userSeq      # 회원 식별 번호 
    problem_id = data.problemId  # 문제 아이디 
    
    logger.info("요청한 유저 : " + str(user_seq))

    # SSE 1 
    message_dto = MessageDTO(progress_info="코드 분석 시작", percentage=25, state="ok", user_seq=user_seq)
    await send_callback("alert", message_dto)

    # 지원 언어, 정답 여부 판단 
    if await filtering_input_data(data, user_seq, send_callback):
        return
    
    
        
    data.language = await parse_lang_type(data.language)
    local_chat_llm_0 = ChatOpenAI(temperature=0, openai_api_key=local_open_ai_api_key, request_timeout=120)

    if data.openai_api_key == "0" :     # 회원 OPEN_AI_API_KEY 없음 
        chat_llm_0 = local_chat_llm_0 
    else :                              # 회원 OPEN_AI_API_KEY 있음  
        decoded_data = b64decode(data.openai_api_key)
        decrypted_api_key = cipher.decrypt(decoded_data)
        api_key = decrypted_api_key.decode().rstrip('\r')
        chat_llm_0 = ChatOpenAI(temperature=0, openai_api_key=api_key, request_timeout=120)

    json_chain = await json_formatter(chat_llm_0)
    
    lock_name = f"problem_id{problem_id}"
    ### 분산락 
    while True:
        try:
            lock = redis_lock.Lock(redis_conn, lock_name, expire=LOCK_TIMEOUT, auto_renewal=True)
            if lock.acquire(blocking=False):
                logger.info("분산락 시작")
                await summary_problem(problem_id, user_seq, data, local_chat_llm_0, json_chain)
                lock.release()
                break
            else:
                logger.info("동일한 문제 대기중")
                await asyncio.sleep(1)
        except Exception as e:
            lock.release()
            logger.error(f"예외 발생: {e}")

            # User 서버에게 실패 이벤트 전송  
            user_service_dto = UserServiceDTO(isSuccess="NO", userSeq=user_seq)
            await send_callback("user-service", user_service_dto)

            raise MyCustomError("에러 발생 - 문제 요약")



    
    # DB에서 문제 요약 정보 조회 
    gpt_problem_summary = await get_gpt_problem_summary(problem_id)

    logger.info("코드 요약 시작")
    
    summary_code_complexity_coroutine = summary_code_complexity(chat_llm_0, data, gpt_problem_summary)
    summary_code_refactor_coroutine = summary_code_refactor(chat_llm_0, data, gpt_problem_summary)
    summary_code_complexity_result, summary_code_refactor_result = await gather(summary_code_complexity_coroutine, summary_code_refactor_coroutine)

    ### SSE 2
    logger.info("코드 요약 완료")
    message_dto = MessageDTO(progress_info="코드 요약 완료", percentage=50, state="ok", user_seq=user_seq)
    await send_callback("alert", message_dto)
    
    logger.info("코드 요약 json 타입으로 변환 시작")
    summary_code_text = f"{OPEN_BRACE}{summary_code_complexity_result}\n {summary_code_refactor_result}\n total_score: 0\n{CLOSE_BRACE}"
    preprocessed_summary_code = await json_chain.arun(data = summary_code_text)
    summary_code_json = await parse_summary_code(chat_llm_0, preprocessed_summary_code)
    logger.info("코드 요약 json 타입으로 변환 완료")

    logger.info("데이터 번역 작업 시작")
    result = await translate_texts(chat_llm_0, summary_code_json)
    logger.info("데이터 번역 작업 완료")

    result.total_score = (result.gpt_solution_time_score + result.gpt_solution_space_score + result.gpt_solution_clean_score) // 3

    ### SSE 3
    message_dto = MessageDTO(progress_info="데이터 번역 완료", percentage=75, state="ok", user_seq=user_seq)
    await send_callback("alert", message_dto)
    
    ### DB 접근 ###
    logger.info("메인 트랜잭션 시작")
    await main_transaction(problem_id, user_seq, data, result, send_callback)
    logger.info("메인 트랜잭션 종료")

    ### SSE 4
    message_dto = MessageDTO(progress_info="코드 분석 완료", percentage=100, state="finish", user_seq=user_seq)
    await send_callback("alert", message_dto)




# 회원 문제 DB 저장 + GPT 응답 DB 저장 + 문제 메타 DB 업데이트 및 저장 
async def main_transaction(problem_id : int, user_seq : int, data : ProblemData, result, send_callback):
    async with async_session() as session:
        try:
            user_submit_problem_data = await save_user_problem_origin(problem_id, user_seq, data.submissionTime, session)
            submission_id = await save_user_submit_solution_origin(problem_id, user_seq, user_submit_problem_data.user_submit_problem_seq, data, session)
            await save_gpt_solution(submission_id, user_seq, result, session)

            lock_name = f"problem_meta{problem_id}"
            ### 분산락 
            while True:
                try:
                    lock = redis_lock.Lock(redis_conn, lock_name, expire=LOCK_TIMEOUT, auto_renewal=True)
                    logger.info("동일한 문제 메타 데이터 대기중")
                    if lock.acquire(blocking=False):
                        logger.info("분산락 시작")
                        await update_problem_meta(problem_id, user_seq, data, session)
                        lock.release()
                        break
                    else:
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"예외 발생: {e}")
                    raise MyCustomError("분산락 에러 발생 - 문제 메타 데이터")

            await session.commit()
        except Exception as e:
            logger.error(f"트랜잭션 처리 중 예외 발생 : {e}")

            # User 서버에게 실패 이벤트 전송  
            user_service_dto = UserServiceDTO(isSuccess="NO", userSeq=user_seq)
            await send_callback("user-service", user_service_dto)

            message_dto = MessageDTO(progress_info="DB 작업 중 예외 발생", percentage=-1, state="error", user_seq=user_seq)
            await send_callback("alert", message_dto)
            raise MyCustomError("트랜잭션 처리 중 에러 발생")


# 문제 정보 요약 함수 
async def summary_problem(problem_id : int, user_seq : int, data : ProblemData, chat_llm, json_chain):
    is_gpt_problem_summary_exist = await check_gpt_problem_summary_is_exist(problem_id)
    if not is_gpt_problem_summary_exist:
        ### 문제 DB에 문제 저장 ### 
        await save_problem_origin(problem_id, data)

        # 문제 요약 정보 생성 
        summary_info_result = await summary_info(chat_llm, data)

        # 문제 요약 정보를 참고하여 모범 답안 생성 (시간 복잡도, 공간 복잡도)
        summary_description_result = await summary_description(chat_llm, data, summary_info_result)
        
        summary_text = f"{OPEN_BRACE}'problem_id': {data.problemId}\n {summary_info_result} \n {summary_description_result}\n{CLOSE_BRACE}" 
        logger.info("문제 요약 정보 전처리")
        preprocessed_summary = await json_chain.arun(data = summary_text)
        summary_json = await parse_summary(chat_llm, preprocessed_summary)

        ### GPT_문제 요약 DB 접근 ###
        await save_problem_summary(problem_id, summary_json)
    
    
async def filtering_input_data(data : ProblemData, user_seq : int, send_callback):
    code_language = await parse_lang_type(data.language)
    if code_language == "unknown":
        logger.info("지원하지 않는 언어")

        # User 서버에게 실패 이벤트 전송  
        user_service_dto = UserServiceDTO(isSuccess="NO", userSeq=user_seq)
        await send_callback("user-service", user_service_dto)

        message_dto = MessageDTO(progress_info="지원하지 않는 언어입니다.", percentage=-1, state="error", user_seq=user_seq)
        await send_callback("alert", message_dto)
        return True
    if data.resultCategory != "ac":
        logger.info("오답 코드")

        # User 서버에게 실패 이벤트 전송  
        user_service_dto = UserServiceDTO(isSuccess="NO", userSeq=data.userSeq)
        await send_callback("user-service", user_service_dto)

        message_dto = MessageDTO(progress_info="틀린 코드는 평가할 수 없습니다", percentage=-1, state="error", user_seq=user_seq)
        await send_callback("alert", message_dto)
        return True