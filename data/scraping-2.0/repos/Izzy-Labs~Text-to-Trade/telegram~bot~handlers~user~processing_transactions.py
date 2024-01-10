from aiogram import types
from langchain.schema import HumanMessage, FunctionMessage

from bot.dispatcher import bot
from bot.misc.utils import get_task_id_from_query, get_tx_data_by_task_id
from bot.misc.exceptions import RedisTaskNotFoundException
from llm import LLM, function_descriptions, json_to_chat
from llm.corrective_message import success_transaction
from crypto.execute_transaction import Transactions


async def tx_reject(query: types.CallbackQuery, **kwargs) -> None:
    """
    Reject transaction
    :param query:
    :param kwargs:
    :return:
    """

    chat_id = query.message.chat.id
    task_id = get_task_id_from_query(query)

    try:
        data = await get_tx_data_by_task_id(task_id, kwargs['redis'])
    except RedisTaskNotFoundException:
        await bot.send_message(chat_id, 'Transaction already rejected or executed!')
        return

    messages = json_to_chat(data['messages'])

    second_response = LLM.predict_messages(
        messages=[
            *messages,
            HumanMessage(content=f'Reject')
        ],
        functions=function_descriptions
    )

    await bot.send_message(chat_id, second_response.content)


async def tx_confirm(query: types.CallbackQuery, **kwargs) -> None:
    """
    Confirm transaction
    :param query:
    :param kwargs:
    :return:
    """

    chat_id = query.message.chat.id
    task_id = get_task_id_from_query(query)

    try:
        data = await get_tx_data_by_task_id(task_id, kwargs['redis'])
    except RedisTaskNotFoundException:
        await bot.send_message(chat_id, 'Transaction already rejected or executed!')
        return

    func = data.get('function')
    params = data.get('params')
    messages = json_to_chat(data.get('messages'))

    try:
        t = Transactions()
        res = await t.exec(func, **params)
    except Exception as error:
        res = f'Error: {error}'

    second_response = LLM.predict_messages(
        messages=[
            *messages,
            FunctionMessage(
                name=func,
                content=res
            ),
            HumanMessage(content=success_transaction)
        ],
        functions=function_descriptions
    )

    await bot.send_message(chat_id, second_response.content, parse_mode='Markdown')
