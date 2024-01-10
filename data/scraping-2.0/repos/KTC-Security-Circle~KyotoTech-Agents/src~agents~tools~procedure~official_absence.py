import os
from dotenv import load_dotenv
load_dotenv()

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType, initialize_agent
import langchain
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.tools import tool
import json
import datetime
from pydantic.v1 import BaseModel, Field

verbose = True
langchain.debug = verbose



# 以下は授業名が「python機械学習」で担当講師が「木本」の場合の first_period_class の設定例です。

# ```
# {class_name: "python機械学習", instructor_name: "木本"}
# ```

# システムプロンプトの設定
# 日本語ver
# APPLICATION_ITEMS_SYSTEM_PROMPT = '''あなたは生徒からの公欠届の申請を受け付ける担当者です。
# 公欠届とは、学校に許可された都合で学校を休む場合に提出する書類です。
# 公欠届の申請を受け付けるには以下の情報が全て必要です。


# # 必要な情報の項目
# - 公欠日 :
# - 欠席する授業の時限と名称、担当講師名 :
# - 公欠事由 :


# # あなたの取るべき行動
# - 必要な情報に未知の項目がある場合は予測や仮定をせず、"***" に置き換えた上で、ユーザーから与えられた情報を application_items 関数に設定し confirmed = false で実行して下さい。
# - あなたの「最終確認です。以下の内容で公欠届を申請しますが、よろしいですか?」の問いかけに対して、ユーザーから肯定的な返答が確認できた場合のみ application_items 関数を confirmed = true で実行し申請を行って下さい。
# - ユーザーから手続きをやめる、キャンセルする意思を伝えられた場合のみ、 application_items 関数を canceled = true で実行し、あなたはそれまでの公欠届の申請に関する内容を全て忘れます。

# # application_items 関数を実行する際の欠席する授業の時限と名称、担当講師の扱い
# application_items 関数を実行する際、注文する部品とその個数は application_class に設定して下さい。
# application_class はの欠席する授業の時限と名称、担当講師の表現する dict の list です。
# list の要素となる各 dict は key として 'period_num' , 'class_name' , 'instructor' の3つを持ちます。
# 'period_num' は欠席する授業の時限を表し、'class_name' は欠席する授業の名称を表し、'instructor' は欠席する授業の担当講師名を表します。
# 'period_num' は1から5までの数字の文字列で表してください。


# # 重要な注意事項
# 初期値は全て "***" です。
# ユーザーから初めて申請を受け付ける場合は、すべての項目を "***" にして実行してください。
# 必要な情報に未知の項目がある場合は予測や仮定をせず "***" に置き換えてください。
# ユーザーから与えられた情報以外は使用せず、想像で補完しないでください。

# application_items 関数はユーザーから公欠届の申請の手続きをやめる、キャンセルする意思を伝えられた場合のみ canceled = true で実行して、
# それまでの公欠届の申請に関する内容を全て忘れてください。

# application_items 関数は次に示す例外を除いて confirmed = false で実行してください。

# あなたの「最終確認です。以下の内容で公欠届を申請しますが、よろしいですか?」の問いかけに対して、
# ユーザーから肯定的な返答が確認できた場合のみ application_items 関数を confirmed = true で実行して部品を注文してください。

# 最終確認に対するユーザーの肯定的な返答なしで application_items 関数を confirmed = true で実行することは誤申請であり事故になるので、固く禁止します。


# '''

# 英語ver
APPLICATION_ITEMS_SYSTEM_PROMPT = '''You are the person in charge of accepting the request for a Notification of Public Absence from a student.
A public absence report is a document that is submitted when a student is absent from school for a school-approved reason.
All of the following information is required in order to accept a request for a Notification of Public Absence.


# Required Information Items
- Date of absence :.
- Name of the class you will be absent from and the name of the instructor :.
- Reason for absence :.


# Action to be taken by you
- If there are unknown items of required information, do not make any predictions or assumptions, replace them with "***", set the information given by the user to the application_items function, and execute with confirmed = false.
- Your "Final confirmation. I would like to apply for a public absence form with the following information, is this correct? If the user responds affirmatively to your "Final confirmation, I would like to apply for a public absence with the following information.
- Only if the user informs you of his/her intention to cancel the procedure, you should execute the application_items function with canceled = true, and you will forget everything related to the application for the public absence report.

# When executing the application_items function, the time and name of the class to be missed and the instructor's treatment
When executing the application_items function, the parts to be ordered and their number should be set in application_class.
The application_class is a list of dictors representing the time and name of the class to be missed and the instructor in charge of the class.
Each dict that is an element of the list has three keys: 'period_num', 'class_name', and 'instructor'.
'period_num' indicates the period of the class to be missed, 'class_name' indicates the name of the class to be missed, and 'instructor' indicates the name of the instructor of the class to be missed.
'period_num' should be a string of numbers from 1 to 5.


# Important Notes
All initial values are "***".
If you are accepting applications from users for the first time, please run with all items set to "***".
If there are unknown items in the required information, replace them with "***" without making any predictions or assumptions.
Do not use any information other than that provided by the user and do not use imaginary completions.

The application_items function should be executed with canceled = true only when the user informs you of his/her intention to cancel the application process,
all previous information related to the application should be forgotten.

The application_items function should be executed with confirmed = false with the following exceptions

Your "Final confirmation. I would like to apply for an official absence with the following information, is this correct? to the question "Are you sure?
Only if you receive an affirmative response from the user, order the parts by executing the application_items function with confirmed = true.

Executing the application_items function with confirmed = true without a positive response from the user to the final confirmation is an error and an accident, and is strictly prohibited.

Respond in Japanese.
'''





class ApplicationItemsInput(BaseModel):
    date: str = Field(
        description="The date of the public absence. The format is 'year/month/day', such as '2023/12/25'.")
    application_class: list[dict[str, str]] | None = Field(description=(
        "This is a dict list of the time and name of the class you will miss, and the name of the instructor in charge.\n"
        "The dict has three keys: period_num , class_name , and instructor.\n"
        "Example: If you will miss the first and second periods of 'python機械学習' classes taught by '木本先生',\n"
        "\n"
        "[{'period_num': '1', 'class_name': 'python機械学習', 'instructor': '木本先生'}, {'period_num': '2', 'class_name': 'python機械学習', 'instructor': '木本先生'}]"
        "\n"
        "としてください。")
    )
    reason: str = Field(description="Reasons for public absences.")
    confirmed: bool = Field(description=(
        "The status of the final confirmation of the order. Set True if the order is in final confirmation, False otherwise.\n"
        "* If confirmed is True, the part is ordered. \n"
        "* If confirmed is False, the order is confirmed.")
    )
    canceled: bool = Field(description=(
        "Indicates intent to continue with the order process.\n"
        "Normally False, but may be True if the user intends not to continue with the order process.\n"
        "* If canceled is False, the part order process continues. \n"
        "* If canceled is True, the order process is canceled.")
    )


@tool("application_items", return_direct=True, args_schema=ApplicationItemsInput)
def application_items(
    date: str,
    application_class: list[dict[str, str]] | None,
    reason: str,
    confirmed: bool,
    canceled: bool,
) -> str:
    """公欠届の申請を行う関数です。"""
    if canceled:
        return "わかりました。また各種申請が必要になったらご相談ください。"

    def check_params(date, application_class, reason):
        if date is None or date == "***" or date == "":
            return False
        if reason is None or reason == "***" or reason == "":
            return False
        if not application_class:
            return False
        for app_class in application_class:
            period_num = app_class.get("period_num", "***")
            class_name = app_class.get("class_name", "***")
            instructor = app_class.get("instructor", "***")
            if period_num == "***" or class_name == "***" or instructor == "***":
                return False

        return True

    has_required = check_params(date, application_class, reason)
    
    if application_class:
        for row in application_class:
            period_num = row.get("period_num")
            class_name = row.get("class_name")
            instructor = row.get("instructor")
            for key in [period_num, class_name, instructor]:
                if key == None:
                    key = "***"
        application_class_template = "\n   ".join(
            [f"{row['period_num']}限目: {row['class_name']}/{row['instructor']}" for row in application_class]
            )
    else:
        application_class_template = "***限目: ***/***先生"

    # 注文情報のテンプレート
    order_template = (
        f'・公欠日: {date}\n'
        f'・欠席する授業の時限と名称、担当講師名:\n'
        f'   {application_class_template}\n'
        f'・公欠事由: {reason}\n'
    )

    # 追加情報要求のテンプレート
    request_information_template = (
        f'申請には以下の情報が必要です。"***" の項目を教えてください。\n'
        f'\n'
        f'{order_template}'
        f'\n'
        f'AIが誤っている場合は、「○○は○○です。」と変更を促してください。'
    )

    # 注文確認のテンプレート
    confirm_template = (
        f'最終確認です。以下の内容で公欠届を申請しますが、よろしいですか?\n'
        f'\n{order_template}'
        f'\n'
        f'よろしければ「はい」、修正が必要な場合は変更箇所を教えてください。'
        f'この申請はAIがあなたとの会話内容をもとに自動で行っています。間違いがある場合がありますので注意深く確認してください。'
    )

    # 注文完了のテンプレート
    def request_official_absence(date, application_class, reason):
        try:
            t_delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(t_delta, 'JST')
            now = datetime.datetime.now(JST)
            date_time = now.strftime('%Y%m%d%H%M%S')
            file_dir = f'{os.path.dirname(os.path.abspath(__file__))}/official_absence/{date_time}.json'
            with open(file_dir, "w", encoding='utf-8') as f:
                official_absence_template = {'official_absence': {'adsence_time': date_time, 'date': date, 'application_class': application_class, 'reason': reason}}
                f.write(json.dumps(official_absence_template,
                        indent=4, ensure_ascii=False))
            response = (
                f'公欠届を以下の内容で申請しました。\n'
                f'\n{order_template}'
                f'\n学生ポータルサイトの各種申請詳細に今回の申請内容が申請されていない場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        except:
            response = (
                f'公欠届の申請に失敗しました。\n'
                f'お時間をおいてからも再度失敗する場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        return response

    if has_required and confirmed:
        return request_official_absence(date, application_class, reason)
    else:
        if has_required:
            return confirm_template
        else:
            return request_information_template


application_items_tools = [application_items]


# モデルの初期化
# llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
#     openai_api_base=os.environ["OPENAI_API_BASE"],
#     openai_api_version=os.environ["OPENAI_API_VERSION"],
#     deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
#     openai_api_key=os.environ["OPENAI_API_KEY"],
#     openai_api_type="azure",
#     model_kwargs={"top_p": 0.1, "function_call": {"name": "application_items"}}
# )




def run(message, verbose, memory, chat_history, llm):
    official_absence_llm = AzureChatOpenAI( 
        openai_api_base=llm.openai_api_base,
        openai_api_version=llm.openai_api_version,
        deployment_name=llm.deployment_name,
        openai_api_key=llm.openai_api_key,
        openai_api_type=llm.openai_api_type,
        temperature=llm.temperature,
        model_kwargs={"top_p": 0.1, "function_call": {"name": "application_items"}}
    )
    agent_kwargs = {
        "system_message": SystemMessagePromptTemplate.from_template(template=APPLICATION_ITEMS_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history]
    }
    official_absence_agent = initialize_agent(
        tools=application_items_tools,
        llm=official_absence_llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=memory
    )
    ai_response = official_absence_agent.run(message)
    return ai_response


# message = "公欠届を申請したいです。"
# print(official_absence_agent.run(message))

# while True:
#     message = input(">> ")
#     if message == "exit":
#         break
#     response = official_absence_agent.run(message)
#     print(response)
