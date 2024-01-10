from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from util.logger import async_log

sql2text_template = """Объясни SQL запрос для топ–менеджера, который не знает, что такое SQL. Не показывай SQL. Уложиcь в 5 предложений

SQL запрос:
{input}

Используй формат в виде нумерованного списка
"""

sql2text_prompt = PromptTemplate(
    template=sql2text_template,
    input_variables=["input"]
)

llm = ChatOpenAI(temperature=0, verbose=False, model_name="gpt-4")

sql2text_chain = LLMChain(llm=llm, prompt=sql2text_prompt, verbose=False)


@async_log("Генерация описания SQL")
async def describe_sql(sql: str) -> str:
    response = await sql2text_chain.arun(sql)
    return response
