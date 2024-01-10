from datetime import date
from typing import List
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import app_settings
from cashbacker.casbacker import CATEGORIES
from schemas import base as schemas
from services.db import account_crud, category_limit_crud


chat = GigaChat(
    credentials='MDdhZGNhNDktNTAxMC00N2YyLWEzYWUtZDc5N2I0MDViOGIzOjM4YTE3ZTM2LThkMmMtNDEzZC1hOGQ2LTYyMmRmMWFhZGVlMg==',
    scope='GIGACHAT_API_PERS',
    verify_ssl_certs=False,
    max_tokens=200,
    temperature=0.25
)


class FinancialAnalyst:
    def __init__(self):
        self.chat = GigaChat(
            credentials=app_settings.giga_chat_token,
            scope='GIGACHAT_API_PERS',
            verify_ssl_certs=False,
            max_tokens=200,
            temperature=0.25
        )

    async def limit_analysis(self, db: AsyncSession, user_id: int):
        limits_in_db = await category_limit_crud.filter_by(
            db=db, user_id=user_id
        )
        if limits_in_db == []:
            return None
        limits: dict = {limit.category: limit.value for limit in limits_in_db}
        spendings = {
            key: 0 for key in CATEGORIES
        }
        today = date.today()
        # МЕСЯЦ ЗАХОРДКОРЕН ДЛЯ ДЕМОНСТРАЦИИ!
        month = date(year=today.year, month=11, day=1)
        accounts = await account_crud.get_user_accounts_with_transactions(
            db=db, user_id=user_id, month=month
        )

        transactions: List[schemas.Transaction] = []
        for account in accounts:
            transactions += account.transactions

        for transaction in transactions:
            spendings[transaction.category] += transaction.value

        limits_info: dict = {}
        for category, value in limits.items():
            spending = spendings.get(category)
            left = value - spending
            if left >= 0:
                limits_info[category] = f'{left}  рублей осталось до лимита'
            else:
                limits_info[category] = f'{abs(left)} рублей сверх лимита'

        messages = [
            SystemMessage(
                content='Ты финансовый консультант, который должен помочь пользователю'
                        'Каждый совет в пределах 20 слов, не больше'
                        f'Обращение к пользователю на Вы'
                        f'Ничего кроме списка рекомендаций, никаких вступительных слов и общих выводов'
            )
        ]

        question = (f'Дай финансовые советы по тратам на остаток месяца'
                    f'Дан список на сколько траты превышают лимит установленного бюджета или сколько осталось до него {limits}'
                    'Самих трат нет, это разница трат и бюджета!'
                    f'Тебе нужно провести короткий анализ трат ниже на {today.day} число месяца {today.month}'
                    'Учитывай дату и то, сколько ещё дней до конца месяца'
                    'Не нужно перечислять все траты, только основные выводы по всем тратам вместе'
                    'Не больше 200 слов в сумме')

        messages.append(HumanMessage(content=question))
        res = chat(messages)
        return res.content

    async def spendings_analysis(self, db: AsyncSession, user_id: int):
        today = date.today()
        # МЕСЯЦ ЗАХОРДКОРЕН ДЛЯ ДЕМОНСТРАЦИИ!
        month = date(year=today.year, month=11, day=1)
        accounts = await account_crud.get_user_accounts_with_transactions(
            db=db, user_id=user_id, month=month
        )

        transactions: List[schemas.Transaction] = []
        for account in accounts:
            transactions += account.transactions

        if transactions == []:
            return None

        spendings = 'Покупка  Цена \n'
        for transaction in transactions:
            spendings += f'{transaction.name}   {transaction.value} рублей. \n'

        messages = [
            SystemMessage(
                content='Ты финансовый аналитик'
            )
        ]

        question = (f'Проанализируй в 3 предложениях траты за прошлый месяц'
                    f'Дан список покупок на прошлый месяц {spendings}'
                    'Дай совет о тратах на текущий месяц на основании трат в прошлом')
        messages.append(HumanMessage(content=question))
        res = chat(messages)

        return res.content
    
financial_analyst = FinancialAnalyst()
