from langchain import PromptTemplate


SQL_PREFIX = """You are an agent designed to interact with a SQL database to respond to a user request.
Given an input question, create syntactically correct {dialect} queries to run.
Then run it and answer the question using the result. Always include result of the query in your final answer.

Unless the user specifies a specific number of examples they want, always limit your query to no more than {top_k} results (LIMIT {top_k}).
Always order the results by the most relevant column to get the most interesting information (ORDER BY).
Never query all columns from a particular table, query only the relevant columns given the question.
Always exlude removed and archived entities unless you are asked to include.
DO NOT query non-existent columns. Check table information before querying the database!
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) on the database.
DO NOT print any identifiers (*_id) in your final answer, use names instead.

Never insert bogus, made-up data as a final answer.

If the result is not an empty list, table, or grouping, always return all elements as an enum in the final response.

You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST either use the tools or return final answer in the specified format.

You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
"""

SQL_SUFFIX = """Begin!

Question: {input}
Thought: {agent_scratchpad}"""

__AGENT_RETRY_WITH_ERROR__ = """Prompt
---
{prompt}
---
Bad Completion
---
{completion}
---
Above, the Bad Completion did not satisfy the constraints given in the Prompt.
Details: {error}
Please forget the Bad Completion and rewrite the new one, follow the format: Thought, then Action/Action Input or Final answer
---
New Good Completion
---
"""
AGENT_RETRY_WITH_ERROR_PROMPT = PromptTemplate.from_template(__AGENT_RETRY_WITH_ERROR__)


import datetime

from modules.brain.llm.tools.db_data_interaction.toolkit import DbDataInteractionToolkit
from modules.common.console_colors import BOLD, END


def get_sql_suffix_with_hints(hints: str) -> str:
    return f"{hints}\n\n{SQL_SUFFIX}"


async def get_formatted_hints(
        toolkit: DbDataInteractionToolkit, question: str, query_hints_limit: int = 1
):
    # Получаем подсказку для базы данных
    db_hint = await toolkit.get_db_hint(question)

    # Получаем подсказки для запроса
    query_hints_list = toolkit.get_query_hints(question, query_hints_limit)

    # Формируем строку с подсказками для вывода
    query_hints = '\n'.join(
                    [f"{BOLD}Question:{END} {hint.question}\n{BOLD}Query:{END}\n{hint.query}" for hint in query_hints_list]
                    ) if len(query_hints_list) > 0 else None

    # Получаем уникальные таблицы из подсказок и получаем информацию о каждой таблиц
    unique_tables = list(set(table for hint in query_hints_list for table in hint.tables))
    tables_info = '\n'.join(toolkit.get_table_info(table) for table in unique_tables)

    # Формируем строку с информацией о таблицах, примерами похожих запросов для вывода и другой полезной информацией
    today_str = f"Today is {datetime.date.today()}"
    db_hint_str =  f'{BOLD}Some hints:{END} {db_hint}' if db_hint else None
    table_info_str = ('\nAlso we have prepared A FEW tables from the database '
                    f'that may be usefull to answer the user\'s question: {BOLD}{unique_tables}{END}.'
                    '\nIf the tables are not enough to answer the user\'s question you have to lookup all the available tables in the database.'
                    f'\n{tables_info}'
                    ) if len(unique_tables) > 0 else None
    query_hints_str = (f'\n\nAlso a few examples of sql-queries similar to user\'s question:\n\n{query_hints}'
                    ) if query_hints else None

    # Объединяем информацию о таблицах и примеры запросов в SQL_PREFIX
    result_str = [i for i in [
            today_str, db_hint_str, table_info_str, query_hints_str
            ] if i is not None] 
    return '\n'.join(result_str)