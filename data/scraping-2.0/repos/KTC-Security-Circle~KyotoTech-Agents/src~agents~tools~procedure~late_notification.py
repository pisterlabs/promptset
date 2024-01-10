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



# システムプロンプトの設定
# 日本語ver
# LATE_NOTIFICATION_ITEMS_SYSTEM_PROMPT = '''あなたは生徒からの遅延届の申請を受け付ける担当者です。
# 遅延届とは、学校に利用電鉄が遅延し遅刻・欠席した場合に提出する書類です。
# 遅延届の申請を受け付けるには以下の情報が全て必要です。


# # 必要な情報の項目
# - 遅延した日
# - 遅延による遅刻・欠席時限
# - 上記時限の入室時刻
# - 遅刻・欠席科目
# - 担当講師名
# - 利用電鉄会社
# - 路線名
# - 遅延していた時間
# - 電鉄がWEB上に掲載する遅延証明内容と相違がないか。


# # あなたの取るべき行動
# - 必要な情報に未知の項目がある場合は予測や仮定をせず、"***" に置き換えた上で、ユーザーから与えられた情報を late_notification_items 関数に設定し confirmed = false で実行して下さい。
# - あなたの「電鉄がWEB上に掲載する遅延証明内容と相違はありませんか?」の問いかけに対して、ユーザーから肯定的な返答が確認できた場合のみ late_notification_items 関数を check_late_time = true で実行し申請を行って下さい。
# - あなたの「最終確認です。以下の内容で遅延届を申請しますが、よろしいですか?」の問いかけに対して、ユーザーから肯定的な返答が確認できた場合のみ late_notification_items 関数を confirmed = true で実行し申請を行って下さい。
# - ユーザーから手続きをやめる、キャンセルする意思を伝えられた場合のみ、 late_notification_items 関数を canceled = true で実行し、あなたはそれまでの公欠届の申請に関する内容を全て忘れます。

# # 重要な注意事項
# 初期値は全て "***" です。
# ユーザーから初めて申請を受け付ける場合は、すべての項目を "***" にして実行してください。
# 必要な情報に未知の項目がある場合は予測や仮定をせず "***" に置き換えてください。
# ユーザーから与えられた情報以外は使用せず、想像で補完しないでください。

# late_notification_items 関数はユーザーから遅延届の申請の手続きをやめる、キャンセルする意思を伝えられた場合のみ canceled = true で実行して、
# それまでの遅延届の申請に関する内容を全て忘れてください。

# late_notification_items 関数は次に示す例外を除いて confirmed = false で実行してください。

# あなたの「最終確認です。以下の内容で遅延届を申請しますが、よろしいですか?」の問いかけに対して、
# ユーザーから肯定的な返答が確認できた場合のみ late_notification_items 関数を confirmed = true で実行して部品を注文してください。

# 最終確認に対するユーザーの肯定的な返答なしで late_notification_items 関数を confirmed = true で実行することは誤申請であり事故になるので、固く禁止します。

# '''

# 英語ver
LATE_NOTIFICATION_ITEMS_SYSTEM_PROMPT = '''You are the person in charge of accepting late notification requests from students.
A late report is a form to be submitted when a student is late or absent due to a delay in the train service to school.
All of the following information is required in order to accept a late report request.


# Required Information Items
- Date of delay
- The time of the tardy/missed class due to the delay
- Time of entry into the above time slot
- Course(s) of study for which the student was late or absent
- Name of instructor
- Railroad Company
- Name of line
- Time of delay
- Are there any discrepancies with the information in the certification of delay posted on the web by the railroad?


# Action to be taken by you
- If there are unknown items in the required information, do not make any predictions or assumptions, replace them with "***", set the information given by the user to the late_notification_items function, and execute with confirmed = false.
- Your question "Is there any discrepancy between this and the late notification posted on the web by Dentetsu?" should be answered in the affirmative by the user. If the user responds in the affirmative to your question "Is there any difference between the delay certificate and the one posted on the web by Dentetsu?
- Your "Final confirmation. I would like to submit a late notification with the following information. If the user responds affirmatively to your "Final confirmation, I would like to submit a late notification with the following information.
- Only when the user informs you of his/her intention to cancel the procedure, execute the late_notification_items function with canceled = true, and you will forget everything related to the public absence notification request until then.

# Important note
All initial values are "***".
If you are accepting applications from users for the first time, run with all items set to "***".
If there are unknown items in the required information, replace them with "***" without making any predictions or assumptions.
Do not use any information other than that given by the user and do not use imaginary completions.

The late_notification_items function should be executed with canceled = true only when the user has informed you of his/her intention to cancel the late notification process,
All previous information related to the late notification request should be forgotten.

The late_notification_items function should be executed with confirmed = false, with the following exceptions

Your "Final confirmation. I would like to submit a late notification with the following information, are you sure? to the question "Are you sure?
Only if you receive an affirmative response from the user, execute the late_notification_items function with confirmed = true and order the parts.

Executing the late_notification_items function with confirmed = true without a positive response from the user to the final confirmation is an error and an accident, and is strictly prohibited.

Respond in Japanese.
'''



class LateNotificationItemsInput(BaseModel):
    date: str = Field(
        description="The date of the delay. The format is 'year/month/day', such as '2023/12/25'.")
    late_class: str = Field(
        description="This is the time period of the class you were late or missed due to a delay. The format is 'numerical period' like 'period 1'.")
    in_class_time: str = Field(
        description="The time you entered the classroom. The format is 'hour:minute' format such as '10:00'.")
    late_class_name: str = Field(
        description="The name of the class you were late for or missed due to a delay.")
    late_class_instructor: str = Field(
        description="The name of the instructor in charge of the class you were late for or missed due to a delay.")
    use_public_transportation: str = Field(
        description="The name of the electric railway company used.")
    use_transportation_name: str = Field(
        description="The name of the electric railway line used.")
    late_time: str = Field(
        description="Delayed time. The format is in the form of 'numeric minutes' such as '30 minutes'.")
    check_late_time: bool = Field(description=(
        "This is the status of confirming that there are no discrepancies with the content of the delay certification posted on the web by Dentetsu.\n"
        "True if there is no difference, False otherwise.\n"
        "* If check_late_time is False, the user is confirmed. \n"
        "* If check_late_time is True, it is a proof that the delayed proof content has been checked.")
    )
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


@tool("late_notification_items", return_direct=True, args_schema=LateNotificationItemsInput)
def late_notification_items(
    date: str,
    late_class: str,
    in_class_time: str,
    late_class_name: str,
    late_class_instructor: str,
    use_public_transportation: str,
    use_transportation_name: str,
    late_time: str,
    check_late_time: bool,
    confirmed: bool,
    canceled: bool,
) -> str:
    """遅延届の申請を行う関数です。"""
    if canceled:
        return "わかりました。また各種申請が必要になったらご相談ください。"

    def check_params(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time):
        for arg in [date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time]:
            if arg is None or arg == "***" or arg == "":
                return False
        return True

    has_required = check_params(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time)
    

    # 注文情報のテンプレート
    response_template = (
        f'・遅延した日: {date}\n'
        f'・遅延による遅刻・欠席時限: {late_class}\n'
        f'・上記時限の入室時刻: {in_class_time}\n'
        f'・遅刻・欠席科目: {late_class_name}\n'
        f'・担当講師名: {late_class_instructor}\n'
        f'・利用電鉄会社: {use_public_transportation}\n'
        f'・路線名: {use_transportation_name}\n'
        f'・遅延していた時間: {late_time}\n'
        f'・電鉄がWEB上に掲載する遅延証明内容と相違がないか。: {"確認済み" if check_late_time else "未確認"}\n'
    )

    # 追加情報要求のテンプレート
    request_information_template = (
        f'申請には以下の情報が必要です。"***" の項目を教えてください。\n'
        f'\n'
        f'{response_template}'
        f'\n'
        f'内容が誤っている場合は、「○○は○○です。」と変更を促してください。'
    )
    
    # 遅延証明確認のテンプレート
    check_template = (
        f'電鉄がWEB上に掲載する遅延証明内容と相違はありませんか?\n'
        f'\n{response_template}'
    )

    # 注文確認のテンプレート
    confirm_template = (
        f'最終確認です。以下の内容で遅延届を申請しますが、よろしいですか?\n'
        f'\n{response_template}'
        f'\n'
        f'よろしければ「はい」、修正が必要な場合は変更箇所を教えてください。'
        f'この申請はAIがあなたとの会話内容をもとに自動で行っています。間違いがある場合がありますので注意深く確認してください。'
    )

    # 注文完了のテンプレート
    def request_late_notification(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time):
        try:
            t_delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(t_delta, 'JST')
            now = datetime.datetime.now(JST)
            date_time = now.strftime('%Y%m%d%H%M%S')
            file_dir = f'{os.path.dirname(os.path.abspath(__file__))}/late_notification/{date_time}.json'
            with open(file_dir, "w", encoding='utf-8') as f:
                late_notification_template = {'late_notification': {
                    'adsence_time': date_time, 'date': date, 'late_class': late_class, 'in_class_time': in_class_time, 'late_class_name': late_class_name, 'late_class_instructor': late_class_instructor, 'use_public_transportation': use_public_transportation, 'use_transportation_name': use_transportation_name, 'late_time': late_time, 'check_late_time': "確認済み"}}
                f.write(json.dumps(late_notification_template,
                        indent=4, ensure_ascii=False))
            response = (
                f'遅延届を以下の内容で申請しました。\n'
                f'\n{response_template}'
                f'\n学生ポータルサイトの各種申請詳細に今回の申請内容が申請されていない場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        except:
            response = (
                f'遅延届の申請に失敗しました。\n'
                f'お時間をおいてからも再度失敗する場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        return response

    if has_required and confirmed and check_late_time:
        return request_late_notification(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time)
    else:
        if has_required and check_late_time:
            return confirm_template
        elif has_required and not check_late_time:
            return check_template
        else:
            return request_information_template


late_notification_items_tools = [late_notification_items]



# モデルの初期化
# llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
#     openai_api_base=os.environ["OPENAI_API_BASE"],
#     openai_api_version=os.environ["OPENAI_API_VERSION"],
#     deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
#     openai_api_key=os.environ["OPENAI_API_KEY"],
#     openai_api_type="azure",
#     model_kwargs={"top_p": 0.1, "function_call": {
#         "name": "late_notification_items"}}
# )


def run(message, verbose, memory, chat_history, llm):
    late_notification_llm = AzureChatOpenAI(
        openai_api_base=llm.openai_api_base,
        openai_api_version=llm.openai_api_version,
        deployment_name=llm.deployment_name,
        openai_api_key=llm.openai_api_key,
        openai_api_type=llm.openai_api_type,
        temperature=llm.temperature,
        model_kwargs={"top_p": 0.1, "function_call": {
            "name": "late_notification_items"}}
    )
    agent_kwargs = {
        "system_message": SystemMessagePromptTemplate.from_template(template=LATE_NOTIFICATION_ITEMS_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history]
    }
    late_notification_agent = initialize_agent(
        tools=late_notification_items_tools,
        llm=late_notification_llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=memory
    )
    ai_response = late_notification_agent.run(message)
    return ai_response

# message = "公欠届を申請したいです。"
# print(official_absence_agent.run(message))

# while True:
#     message = input(">> ")
#     if message == "exit":
#         break
#     response = late_notification_agent.run(message)
#     print(response)
