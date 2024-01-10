from langchain.chat_models import ChatOpenAI
from myclass.usercode import UsercodeSummary
from chain.translator.translate_text import translate_text
from asyncio import gather
from logging import getLogger

# logger 설정 
logger = getLogger()

async def translate_texts(chat_llm : ChatOpenAI, data : UsercodeSummary):
    gpt_solution_time_complexity_reason_translator = translate_single_text(chat_llm, data.gpt_solution_time_complexity_reason)
    gpt_solution_time_complexity_good_point_translator = translate_single_text(chat_llm, data.gpt_solution_time_complexity_good_point)
    gpt_solution_time_complexity_bad_point_translator = translate_single_text(chat_llm, data.gpt_solution_time_complexity_bad_point)
    gpt_improving_time_complexity_suggestion_translator = translate_single_text(chat_llm, data.gpt_improving_time_complexity_suggestion)
    gpt_solution_space_complexity_reason_translator = translate_single_text(chat_llm, data.gpt_solution_space_complexity_reason)
    gpt_solution_space_complexity_good_point_translator = translate_single_text(chat_llm, data.gpt_solution_space_complexity_good_point)
    gpt_solution_space_complexity_bad_point_translator = translate_single_text(chat_llm, data.gpt_solution_space_complexity_bad_point)
    gpt_improving_space_complexity_suggestion_translator = translate_single_text(chat_llm, data.gpt_improving_space_complexity_suggestion)
    gpt_solution_refactoring_suggestion_translator = translate_single_text(chat_llm, data.gpt_solution_refactoring_suggestion)
    
    data.gpt_solution_time_complexity_reason, \
    data.gpt_solution_time_complexity_good_point, \
    data.gpt_solution_time_complexity_bad_point, \
    data.gpt_improving_time_complexity_suggestion, \
    data.gpt_solution_space_complexity_reason, \
    data.gpt_solution_space_complexity_good_point, \
    data.gpt_solution_space_complexity_bad_point, \
    data.gpt_improving_space_complexity_suggestion, \
    data.gpt_solution_refactoring_suggestion = await gather(
        gpt_solution_time_complexity_reason_translator,
        gpt_solution_time_complexity_good_point_translator,
        gpt_solution_time_complexity_bad_point_translator,
        gpt_improving_time_complexity_suggestion_translator,
        gpt_solution_space_complexity_reason_translator,
        gpt_solution_space_complexity_good_point_translator,
        gpt_solution_space_complexity_bad_point_translator,
        gpt_improving_space_complexity_suggestion_translator,
        gpt_solution_refactoring_suggestion_translator
    )
    logger.info(data)
    return data

async def translate_single_text(chat_llm : ChatOpenAI, text: str):
    chain = await translate_text(chat_llm)
    translate_result = await chain.arun(text = text)
    return translate_result