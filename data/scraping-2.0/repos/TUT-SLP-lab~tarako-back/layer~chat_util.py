import json
import os

import openai
from table_utils import json_dumps

openai.api_key = os.environ["OPENAI_API_KEY"]
CHATGPT_MODEL = "gpt-4"


def gen_create_task_prompt(msg):
    return (
        """
次の内容のタスクを作成してください。
もし、似た内容のタスクがあった場合、それも教えてください。
（例）
入力 : "A社への請求書作成を開始しました。"
出力 : {
  "title": "A社への請求書作成",
  "category": "Accounting",
  "tags": ["請求書", "A社"],
  "progresses": 0,
  "serious": 0,
  "details": "A社への請求書を作成する",
  "response_message": "A社への請求書作成のタスクを作成しました。"
}

入力 : "会議を始める"
出力 : {
  "title": "会議",
  "category": "GeneralAffairs",
  "tags": ["会議"],
  "progresses": 0,
  "serious": 0,
  "details": "会議を始める",
  "response_message": "会議のタスクを作成しました。"
}

入力 : "B社との会議が長引いている"
出力 : {
  "title": "B社との会議",
  "category": "GeneralAffairs",
  "tags": ["会議"],
  "progresses": 50,
  "serious": 3
  "details": "B社との会議が長引いている",
  "response_message": "B社との会議のタスクについての進捗を更新しました。頑張ってください。"
}

入力 : "C社への電話連絡を完了しました。"
出力 : {
    "title": "C社への電話連絡",
    "category": "GeneralAffairs",
    "tags": ["電話", "C社"],
    "progresses": 100,
    "serious": 0,
    "details": "C社への電話連絡を完了しました。",
    "response_message": "C社への電話連絡のタスクを完了しました。お疲れ様でした。"
    }

"""
        + f"""
```
{msg}
```
"""
    )


def create_task(gpt_output):
    gpt_output = json.loads(gpt_output)
    return {
        "title": gpt_output.get("title"),
        "category": gpt_output.get("category"),
        "tags": gpt_output.get("tags"),
        "progres": gpt_output.get("progress"),
        "serious": gpt_output.get("serious"),
        "details": gpt_output.get("details"),
    }


def create_task_function(category_list):
    return {
        "name": "create_task",
        "description": "タスクオブジェクトを作成する",
        "parameters": {
            "type": "object",
            "properties": {
                "gpt_output": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "タスク名"},
                        "category": {
                            "type": "string",
                            "description": "タスク名のカテゴリ",
                            "enum": category_list,
                        },
                        "tags": {
                            "type": "array",
                            "description": "タスクのタグ(複数可)",
                            "items": {"type": "string"},
                        },
                        "progress": {
                            "type": "integer",
                            "description": "タスクの進捗(0~100)",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "serious": {
                            "type": "integer",
                            "description": "タスクの深刻度(0~5)",
                            "minimum": 0,
                            "maximum": 5,
                        },
                        "details": {
                            "type": "string",
                            "description": "タスクの詳細。タイトルでは表現できない内容を記述する",
                        },
                        "response_message": {
                            "type": "string",
                            "description": "タスクを作成したことをユーザに伝えるメッセージ",
                        },
                    },
                    "required": ["title", "category", "tags", "progress", "serious", "details", "response_message"],
                }
            },
            "required": ["gpt_output"],
        },
    }


def suggest_similer_task_function(task_title_dict: dict[str, str]):
    return {
        "name": "suggest_similer_task",
        "description": "似たようなタスクがあった場合、タスクタイトルとIDのペアを返す",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "object",
                    "description": "タスクオブジェクト",
                    "properties": {
                        "title": {"type": "string", "description": "タスク名"},
                        "task_id": {"type": "string", "description": "タスクID"},
                    },
                },
            },
            "required": ["title", "task_id"],
        },
    }


def gen_task_data(msg: str, category_list: list[str], task_title_dict: dict[str, str] = {}):
    response = openai.ChatCompletion.create(
        temperature=0,
        model=CHATGPT_MODEL,
        messages=[
            {"role": "user", "content": gen_create_task_prompt(msg)},
        ],
        functions=[
            create_task_function(category_list),
            # suggest_similer_task_function(task_title_dict),
        ],
        function_call={"name": "create_task"},
    )
    # if "function_call" not in response["choices"][0]["message"]:
    #     raise FunctionCallingError("function_callがありません")
    task = None
    suggest_task = None
    if response["choices"][0]["message"]["function_call"]["name"] == "create_task":
        task_str = response["choices"][0]["message"]["function_call"]["arguments"]
        task = json.loads(task_str)["gpt_output"]
    elif response["choices"][0]["message"]["function_call"]["name"] == "suggest_similer_task":
        suggest_task_str = response["choices"][0]["message"]["function_call"]["arguments"]
        suggest_task = json.loads(suggest_task_str)
    return task, suggest_task


def gen_create_user_diary_prompt(msg, task_dict: dict[str, str]):
    prompt_task_dict = []
    for task in task_dict:
        prompt_task_dict.append(
            {
                "title": task["title"],
                "category": task["category"],
                "tags": task["tags"],
                "progress": task["progresses"][-1],
                "serious": task["serious"],
                "details": task["details"],
            }
        )
    return f"""
次の内容は事務員の今日タスクです。これらの内容から日報を作成してください。
```
{json_dumps(prompt_task_dict)}
```
最後に従業員の一言です．
{msg}
"""


def create_user_diary_function():
    return {
        "name": "create_diary",
        "description": "複数タスクの情報から日報を作成する",
        "parameters": {
            "type": "object",
            "properties": {
                "details": {
                    "type": "string",
                    "description": "日報の詳細。タスクの情報をできるだけ網羅できていて，" + "分かりやすい内容でMarkdown形式で記述する．",
                },
                "ai_analysis": {
                    "type": "string",
                    "description": "メッセージやタスクからAIが自動で分析した内容を記述する．" + "思いやりがあって，従業員がやる気になるような内容を記述する．",
                },
                # "serious": {
                #     "type": "string",
                #     "description": "日報の深刻度(0~5)を整数のみで記述する．",
                # },
            },
            "required": ["details", "ai_analysis", "serious"],
        },
    }


def gen_user_diary_data(msg: str, task_dict: dict[str, str] = {}):
    response = openai.ChatCompletion.create(
        model=CHATGPT_MODEL,
        messages=[
            {"role": "user", "content": gen_create_user_diary_prompt(msg, task_dict)},
        ],
        functions=[
            create_user_diary_function(),
        ],
    )

    if "function_call" not in response["choices"][0]["message"]:
        raise FunctionCallingError("function_callがありません")
    diary = None
    if response["choices"][0]["message"]["function_call"]["name"] == "create_diary":
        diary_str = response["choices"][0]["message"]["function_call"]["arguments"]
        diary = json.loads(diary_str)
    return diary


def gen_create_section_diary_prompt(user_diary_dict: dict[str, str]):
    prompt_user_diary_dict = []
    for user_diary in user_diary_dict:
        prompt_user_diary_dict.append(
            {
                "details": user_diary["details"],
                "ai_analysis": user_diary["ai_analysis"],
                "serious": user_diary["serious"],
            }
        )
    return f"""
次の内容は課に所属している事務員の今日の日報です。これらの内容から課の日報を作成してください。
```
{json_dumps(prompt_user_diary_dict)}
```
"""


def create_section_diary_function():
    return {
        "name": "create_section_diary",
        "description": "複数人の日報の情報から課の日報を作成する",
        "parameters": {
            "type": "object",
            "properties": {
                "details": {
                    "type": "string",
                    "description": "日報の詳細。タスクの情報をできるだけ網羅できていて，分かりやすい内容でMarkdown形式で記述する．",
                },
                "ai_analysis": {
                    "type": "string",
                    "description": "事務員全体の日報からAIが自動で分析した内容を記述する．課長が見て，部署内の状況が分かるような内容を記述する．",
                },
            },
            "required": ["details", "ai_analysis"],
        },
    }


def gen_section_diary_data(user_diary_dict: dict[str, str] = {}):
    response = openai.ChatCompletion.create(
        model=CHATGPT_MODEL,
        messages=[
            {"role": "user", "content": gen_create_section_diary_prompt(user_diary_dict)},
        ],
        functions=[
            create_section_diary_function(),
        ],
    )
    if "function_call" not in response["choices"][0]["message"]:
        raise FunctionCallingError(f"function_callがありません{response}")
    diary = None
    if response["choices"][0]["message"]["function_call"]["name"] == "create_section_diary":
        diary_str = response["choices"][0]["message"]["function_call"]["arguments"]
        diary = json.loads(diary_str)
    return diary


class FunctionCallingError(Exception):
    pass
