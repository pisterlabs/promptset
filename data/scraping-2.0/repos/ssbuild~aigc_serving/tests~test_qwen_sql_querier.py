import json
import sqlite3
import openai


# 新版本
openai.api_key = "112233"
openai.api_base = "http://192.168.2.180:8081/v1"
openai.api_base = "http://106.12.147.243:8082/v1"
model = "chatglm2-6b-int4"
model = "qwen-7b-chat-int4"
model = "Qwen-14B-Chat"

def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results


class SqlQuerier:
    def __init__(self, db_path="Chinook.db"):

        self.conn = sqlite3.connect(db_path)
        print("Opened database successfully")

    def run(self, query, database_schema):
        # Step 1: send the conversation and available functions to model
        functions = [
            {
                "name_for_human":
                    "数据库查询工具",
                "name_for_model":
                    "ask_database",
                "description_for_model":
                    "该工具用来回答音乐相关的问题，输出应该是一个标准化的SQL查询语句。",
                "parameters": [
                    {
                        'name': 'query',
                        'description': f"基于下面数据库表结构的SQL查询语句，用来回答用户问题。\n\n{database_schema}",
                        'required': True,
                        'schema': {
                            'type': 'string'
                        },
                    },
                ],
            }
        ]

        messages = [{"role": "user",
                     "content": query,}]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            functions=functions,
            stop=["Observation:", "Observation"]
        )


        print(response["choices"][0]["message"]["content"])


        answer = ""
        response_message = response["choices"][0]["message"]
        # Step 2: check if model wanted to call a function
        if response_message.get("function_call"):
            print(f"Function call: {response_message['function_call']}")
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "ask_database": ask_database,
            }  # only one function in this example
            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            print(f"Function args: {function_args}")

            function_response = fuction_to_call(self.conn, function_args["query"])
            print(f"Function response: {function_response}")

            # Step 4: send the info on the function call and function response to model
            messages.append({"role": "assistant","content": response.choices[0].message.content})  # extend conversation with assistant's reply
            messages.append(
                {
                    "role": "function",
                    "content": function_response,
                }
            )  # extend conversation with function response

            second_response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                functions=functions,
            )  # get a new response from model where it can see the function response
            answer = second_response["choices"][0]["message"]["content"]
            print(f"Model output: {answer}")

            j = answer.rfind("Final Answer:")
            answer = answer[j + 14:] if answer else answer

        return answer


if __name__ == '__main__':
    database_schema = """create table albums
AlbumId INTEGER not null primary key autoincrement, --专辑ID
Title NVARCHAR(160) not null, --专辑名称
ArtistId INTEGER not null references artists --艺术家ID
);
"""


    query = "发行专辑最多的艺术家是谁？"
    sql_querier = SqlQuerier()
    answer = sql_querier.run(query, database_schema)
    print(answer)
