import threading

import openai
from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import SQLDatabaseSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase

from config import API_KEY, DB_URI

openai.api_key = API_KEY


class TimeoutException(Exception):
    pass


def run_func(func, args=(), kwargs={}):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print("Exception: ", e)


def timeout_handler(timeout):
    raise TimeoutException("Execution timed out")


# 设置超时时间为 100 秒
timeout = 100

# 创建定时器
timer = threading.Timer(timeout, timeout_handler, [timeout])


def get_completion(prompt, model="gpt-3.5-turbo"):
    print("prompt" + prompt)
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def ques2data(questions):
    db = SQLDatabase.from_uri(DB_URI)
    llm = OpenAI(openai_api_key=API_KEY, temperature=0)
    db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)

    toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0, openai_api_key=API_KEY))
    agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=API_KEY),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS
    )

    answer = ""

    # try:
    #     # 启动定时器
    #     timer.start()

    for question in questions:
        print(question)
        answer += agent_executor.run(question) + " "
        # answer += db_chain.run(question) + " "

    # except TimeoutError:
    #     # 超时处理
    #     print("Timed out")
    # finally:
    #     # 关闭超时
    #     timer.cancel()
    # return answer


def ques2answer(question):
    db = SQLDatabase.from_uri(DB_URI)
    llm = OpenAI(openai_api_key=API_KEY, temperature=0)

    db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)
    return db_chain.run(question)


if __name__ == '__main__':
    print(ques2answer("成交金额最高的的供应商是___，营业额达到____元。他比第二名高出___%。"))
    # prompt = f"""
    # Generate a sql to get average score from table students.
    # """
    # # response = get_completion(prompt)
    # # print(response)
    # template = "共有3种不同的产品，其中最贵的是desa，它来自美国的BR234供应商。"
    # # template = "2022年，演示单位累计实现营业收入总额 231,659.69 万元，其中供应商32342的营业收入最高，较平均值高出30%。"
    #
    # # ask(template)
    # ct = Chat()
    # answer = ct.ask1("\""+
    #     template + "\"请针对以上信息进行提问，要求拆分问题时，保留上下文的条件关系。\n说明：例如‘共有4个订单，其中3个订单的创建时间在今天之前，物品总价格为200美元，他们来自2"
    #              "个不同的消费者’转换为‘有多少订单？其中有多少是在今天之前创建的？这些的总价是多少？这些订单来自多少个不同的客户？’每一个问题应当与句子中第一个对他产生影响的条件相关联。"
    #              "例如‘他们来自两个不同的消费者’应转换为”这些订单来自几个不同的消费者‘或’创建时间在今天之前的订单来自几个不同的消费者‘而不是’物品总价格为200美元的订单"""
    #              "来自几个不同的消费者‘。\n请注意，你只需要识别句子中的信息并给出提问，不需要提问不存在的信息")
    # print(answer)

    # messages = [
    #     {'role': 'system', 'content': 'You are an assistant that speaks like Shakespeare.'},
    #     {'role': 'user', 'content': 'tell me a joke'},
    #     {'role': 'assistant', 'content': 'Why did the chicken cross the road'},
    #     {'role': 'user', 'content': 'I don\'t know'}]
    # response = get_completion_from_messages(messages, temperature=1)
    # print(response)
