from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import AgentType, create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from app.util.responses import ImageResponse, TextResponse, ButtonResponse
from quickchart import QuickChart
from app.config.database import SessionLocal, engine
import os
import decimal
from datetime import datetime
from logging import getLogger
from app.core.GPT_request import get_completion
from app.models.message import Message
from app.config.config import CHARTS, CHARTS_CONFIGS
from sqlalchemy import text
import openai
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.llms.openai import OpenAI
from langchain.tools.json.tool import JsonSpec
import ast
import asyncio
from decimal import Decimal
from app.config.variables import session as cache
from enum import Enum
from app.enums.status_enum import StatusEnum

logger = getLogger("app")

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_CHAT_MODEL"] = "gpt-3.5-turbo-16k-0613"

llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"),
                 temperature=0)

db = SQLDatabase(engine)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm, handle_parsing_errors="Check your output and make sure it conforms!",
                                 return_intermediate_steps=True)
sql_toolkit.get_tools()

sqldb_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True},
    verbose=False,

    format_instructions="""Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question""",
)
message = [{"role": "system",
            "content": "You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about Students, Teachers and Homeworks."},
           {"role": "user", "content": "How many practices did my students do in the last year?\n ai: "}]
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True, return_intermediate_steps=True)

DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 300
DEFAULT_RATIO = 2.0
dashboard_key = "dashboard_key"


class default_chart(QuickChart):
    def __init__(self):
        super().__init__()
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.device_pixel_ratio = DEFAULT_RATIO


def decimal_to_float(data):
    if isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, list):
        return [decimal_to_float(item) for item in data]
    return data


def summary(query, sql, data):
    summary_prompt = "You are an intelligent educational chatbot who can query the database, and you have executed " + str(
        sql) + " and got response " + str(
        data) + "summary the infomations as brief as possile, make it sounds like a {Role} {Action} {Thing}, and {Performance}"
    print(summary_prompt)
    print(query)
    res = get_completion(prompt=query, sys_prompt=summary_prompt, model="gpt-4", temperature=0.2)
    return res


def dispatcher(functions, prompt):
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": prompt}]
    functions = functions
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_message = response["choices"][0]["message"]
    print(response_message)
    if response_message.get("function_call"):
        function_id = response_message["function_call"]["name"]
        name_dict = {
            "Bar_Chart": 0,
            "Line_Chart": 1,
            "Pie_Chart": 2,
            "Radar_Chart": 3,
            "Scatter_Chart": 4,
        }
        config = CHARTS_CONFIGS[name_dict[function_id]]
        return config
        # function_to_call = available_functions[function_name]

        # return function_to_call
    return None


def dashboard(message: Message):
    bot = cache[message.fromNo]
    status = InnerStatus.AFTER
    if bot.inner_status is None or bot.inner_status != InnerStatus.AFTER:
        status = InnerStatus.FIRST
    if status == InnerStatus.FIRST:
        bot.inner_status = InnerStatus.AFTER
        return ButtonResponse(
            "Welcome to dashboard 🎉, you can query in natural languages in dashboard, try some examples: " + \
            "\n\nA. How many practices and daily reads did Emma Johnson do in the last year" + \
            "\nB. What kinds of ability that John Smith have? " + \
            "\nC. How many quizzed and daily reads did Emma Johnson do in the last year",
            ["A", "B", "C"])
    if message.text == "A":
        message.text = "How many practices and daily reads did Emma Johnson do in the last year"
    elif message.text == "B":
        message.text = "What kinds of ability that John Smith have?"
    elif message.text == "C":
        message.text = "Show me the total sum of both number of quizzed and num ber of daily reads that Emma Johnson do in the last year to draw a pie chart."
    # else:
    #     bot.inner_status = InnerStatus.FIRST
    #     return TextResponse("You did something wrong")
    try:
        openai.api_key = "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"
        message.text = message.text + " return in list"
        result = db_chain(message.text)
        sql = result["intermediate_steps"][2]['sql_cmd']
        prompt = get_completion(
            prompt=message.text + ', what kind of graph should I use to show the data? choose from [bar chart, line chart, pie chart, radar chart]. Only respond with one type of chart, do not include any explanation.',
            sys_prompt="You are an intelligent data analyst", model="gpt-3.5-turbo", temperature=0)
        result["intermediate_steps"][2]['sql_cmd'].split("FROM")[0].split("SELECT")[1].split(",")
        session = SessionLocal()
        sql_result = session.execute(text(sql)).fetchall()
        if len(sql_result) == 0:
            return TextResponse("Sorry, there is no data for your query, please try again.")
        dict_name = sql.split("FROM")[0].split("SELECT")[1].split(",")
        new_dict_name = []
        target_dict = {}
        try:
            for index, key in enumerate(dict_name):
                if "AS" in key:
                    key = key.split("AS")[1]
                key = key.split(".")[-1]
                key = key.replace("\n", " ")
                key = key.strip()
                new_dict_name.append(key)
                target_dict[key] = []
                for item in sql_result:
                    target_dict[key].append(item[index])
            print(target_dict)
            target_chart_config = dispatcher(CHARTS, prompt)
            chart = default_chart()
            result = decimal_to_float(result)
            if target_chart_config['type'] == "bar" or target_chart_config['type'] == "line" or target_chart_config[
                'type'] == "scatter":
                json_spec = JsonSpec(dict_=target_dict, max_value_length=4000)
                json_toolkit = JsonToolkit(spec=json_spec)
                print(json_toolkit)
                json_agent_executor = create_json_agent(
                    llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
                )
                result = json_agent_executor.run(
                    "Output all the time in the form of list.",
                )
                result = ast.literal_eval(result)
                print(dict_name)
                if not isinstance(result[0], int):
                    result_str = [datetime(*item).strftime('%Y-%m-%d') for item in result]
                    target_chart_config["data"]["labels"] = result_str
                    for key in target_dict.keys():
                        target_dict[key] = decimal_to_float(target_dict[key])
                        if not target_dict[key][0] == datetime(*result[0]):
                            target_chart_config["data"]["datasets"].append({"label": key, "data": target_dict[key]})
                else:
                    target_chart_config["data"]["labels"] = result
                    for key in target_dict.keys():
                        target_dict[key] = decimal_to_float(target_dict[key])
                        if not target_dict[key][0] == result[0]:
                            target_chart_config["data"]["datasets"].append({"label": key, "data": target_dict[key]})
            elif target_chart_config['type'] == "pie":
                result_str = new_dict_name
                target_chart_config["data"]["labels"] = result_str
                target_chart_config["data"]["datasets"].append({"data": []})
                for key in target_dict.keys():
                    target_dict[key] = decimal_to_float(target_dict[key])
                    target_chart_config["data"]["datasets"][0]["data"].append(target_dict[key])
            elif target_chart_config['type'] == "radar":
                result_str = new_dict_name
                target_chart_config["data"]["labels"] = result_str
                target_chart_config["data"]["datasets"].append({"data": []})
                for key in target_dict.keys():
                    target_dict[key] = decimal_to_float(target_dict[key])
                    target_chart_config["data"]["datasets"][0]["data"].append(target_dict[key])
                    target_chart_config["data"]["datasets"][0]["label"] = "Abilities"
            logger.debug(target_chart_config)
            chart.config = target_chart_config
            print(target_chart_config)
            print(chart.get_url())
            return ImageResponse(chart.get_url() + '.png', text=summary(message.text, sql, target_dict))
        except Exception as e:
            print(e)
            return TextResponse(
                summary(message.text + "Sorry, I do not understand your question, please try again.", sql, target_dict))
    except Exception as e:
        return ButtonResponse(
            "Sorry, I do not understand your question, please try again: " + \
            "\n\nA. How many practices and daily reads did Emma Johnson do in the last year" + \
            "\nB. What kinds of ability that John Smith have? " + \
            "\nC. How many quizzed and daily reads did Emma Johnson do in the last year",
            ["A", "B", "C"])


def dashboard_exit(message):
    bot = cache[message.fromNo]
    bot.main_status = StatusEnum.BEGIN
    bot.inner_status = None


class InnerStatus(Enum):
    FIRST = 1
    AFTER = 2


if __name__ == "__main__":
    # message = Message(text="How many practices and daily reads did Emma Johnson do in the last year, return in list")
    res = dashboard("What kinds of ability that John Smith have?")
    res.change_recipient("6583869990")
    asyncio.run(res.send())