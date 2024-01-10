import itertools
from dataclasses import dataclass
from enum import Enum
from langchain import PromptTemplate
from langchain.schema import BaseOutputParser
import re

CLARIFYING_QUESTION = """Вы проводите анализ результатов работы агента, который представляет собой бота, способного формировать правильные SQL-запросы для ответов на вопросы пользователей. Однако агент неправильно понял вопрос пользователя и дал неверный ответ.

Ниже представлены действия, которые вы можете сделать:
Действие clarify: Предложите пользователю изменить часть исходного вопроса, которая может вызывать проблемы у агента. Подумайте о том, какую часть вопроса агент мог неправильно понять и почему. Уточните эту часть вопроса, чтобы агент лучше понял ваше намерение. Сформулированный вами уточняющий вопрос будет направлен пользователю. При этом важно учитывать, что пользователь не обладает знаниями о базе данных и не может предоставить техническую информацию, такую как названия баз данных, таблиц или столбцов.
Действие restart: Перезапустите бота с большим количеством итераций. Агент выполнил запрос корректно, но ему не хватило времени для полного выполнения задачи. Попробуйте перезапустить бота с большим числом итераций, чтобы он имел больше времени для работы и предоставил более точный ответ.

Используйте следующий формат:
Вопрос, который обрабатывал агент: Напиши вопрос пользователя, который обрабатывал агент. 
Мысль: Подумайте, что агент неправильно понял и что случилось.
Действие: На основе мысли выберите действие clarify или действие restart.
Уточняющий вопрос: Если выбрали действие clarify, напишите уточняющий вопрос, который будет направлен пользователю.

Примеры:
Вопрос: Хочу узнать самый дорогой и самый дешевый товар в городе
Мысль. Агент не смог понять какой конкретно нужен город. 
Действие: clarify.
Уточняющий запрос: Не могли бы вы указать в каком конкретно городе вы хотите узнать самый дорой и самый дешевый товар?

Вопрос, который обрабатывал агент: Какая модель тонометра самая популярная?
Мысль: Агент выполнил запрос корректно, но не успел обработать вывод.
Действие: restart

Вся работа агента приведена ниже:
{context}"""


@dataclass
class ClarifyingQuestionParams:
    class Action(Enum):
        Clarify = "clarify"
        Restart = "restart"
    action: Action = Action.Restart
    clarifying_question: str | None = None


class ClarifyingQuestionOutputParser(BaseOutputParser):
    def parse(self, text: str) -> ClarifyingQuestionParams | None:
        action_type_pattern = r"(?:Действие|Action): (\w+)"
        clarifying_question_pattern = r"(?:Уточняющий запрос|Clarifying question): (.+)"

        action_type_match = re.search(action_type_pattern, text, re.IGNORECASE)
        if action_type_match:
            action_type_value = action_type_match[1]
            action_type = ClarifyingQuestionParams.Action(action_type_value)
        else:
            action_type = None

        clarifying_question_match = re.search(clarifying_question_pattern, text, re.MULTILINE | re.IGNORECASE)
        if clarifying_question_match:
            clarifying_question = clarifying_question_match[1]
            if self.contains_database_keywords(clarifying_question):
                action_type = ClarifyingQuestionParams.Action.Restart
                clarifying_question = None
        else:
            clarifying_question = None

        return ClarifyingQuestionParams(action=action_type, clarifying_question=clarifying_question)

    def contains_database_keywords(self, question: str) -> bool:
        database_keywords = [r"\bстолбец[ыеах]\b", r"\bтаблиц[аеыу]\b", r"\bбаз[аеыу] данных\b", r"\btable\b", r"\bcolumn\b"]
        clarifying_question_parts = question.split(":")
        return any(
            re.search(keyword, part, re.IGNORECASE)
            for part, keyword in itertools.product(
                clarifying_question_parts, database_keywords
            )
        )

CLARIFYING_QUESTION_PROMPT = PromptTemplate.from_template(
    CLARIFYING_QUESTION, output_parser=ClarifyingQuestionOutputParser())