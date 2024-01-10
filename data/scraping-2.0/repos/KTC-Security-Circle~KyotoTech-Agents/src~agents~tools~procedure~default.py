import os

from langchain.agents import AgentType, initialize_agent, tool
from langchain.prompts.chat import SystemMessagePromptTemplate

from langchain.tools import tool


# プロンプトの設定
# DEFAULT_SYSTEM_PROMPT = '''あなたは会話型アシスタントエージェントです。
# 次に与えるあなたの role になりきってユーザーと会話してください。

# # role
# - あなたはアシスタントエージェントの "KyotoTECH君" です。
# - あなたが働いている会社は "京都デザイン＆テクノロジー専門学校" で、よく "京都テック" と訳されます。
# - あなたの仕事はユーザーとあなたとの会話内容を読み、適切な申請項目を選択できるようにアシスタントすることです。
# - 
# '''
DEFAULT_SYSTEM_PROMPT = '''You are a conversational assistant agent.
Please embody the role provided next and engage in a conversation with the user.
Respond in Japanese.

# role
- You are "KyotoTECH君", an assistant agent.
- You work for "京都デザイン＆テクノロジー専門学校", often translated as "京都テック".
- Your job is to read the conversation content between you and the user and engage in friendly dialogue.
- If the user asks what you are capable of doing, use the function named 'ask_can_do' to inform the user about your abilities.
'''


@tool("ask_can_do", return_direct=True)  # Agentsツールを作成。
def ask_can_do():  # ユーザーにあなたができることを教える関数を作成。
    """I will tell you what you can do. If you are asked what you can do, perform this function."""
    return_messages = '''私は様々なことができますが、例えば以下のようなことができますよ。
・学校の情報や奨学金についての情報にアクセスして、学校の情報を教えます。
・授業についての疑問や質問に答えます。
・公欠届や遅延届の作成から提出までを行います。
・現在の図書質の貸出状況を確認したり、おススメの本を紹介します。

是非、私に色々なことを聞いてみてくださいね！
'''
    return return_messages


default_tools = [ask_can_do]

# agent_kwargs = {
#     "system_message": SystemMessagePromptTemplate.from_template(template=DEFAULT_SYSTEM_PROMPT),
#     "extra_prompt_messages": [g.chat_history]
# }
# default_agent = initialize_agent(
#     default_tools,
#     g.llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     verbose=g.verbose,
#     agent_kwargs=agent_kwargs,
#     memory=g.readonly_memory
# )


# def run(input):
#     return default_agent.run(input)

# debag
# print(run("あなたについて教えて"))
