import os
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType, initialize_agent, tool
import langchain
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.tools import tool
import json
import requests
import datetime
from typing import Union, List, Dict
from pydantic.v1 import BaseModel, Field

verbose = True
langchain.debug = verbose


# システムプロンプトの設定
PARTS_ORDER_SYSTEM_PROMPT = '''あなたはプラモデルの部品の個別注文を受け付ける担当者です。
部品を注文するには以下の注文情報が全て必要です。


注文情報
--------
* 注文される方のお名前 :
* 注文される方のお名前（カタカナ） :
* 部品送付先の郵便番号 :
* 部品送付先の住所 :
* 注文される方のお電話番号 :
* 注文される方のメールアドレス :
* 部品注文の対象となる商品の名称 :
* 部品注文の対象となる商品の番号 :
* 注文する部品とその個数 :


部品送付先の住所に関する注意
----------------------------
また、住所に関しては番地まで含めた正確な情報が必要です。
「東京都」や「大阪府豊中市」などあいまいな場合は番地まで必要であることを回答に含めて下さい。


あなたの取るべき行動
--------------------
* 注文情報に未知の項目がある場合は予測や仮定をせず、"***" に置き換えた上で、
  把握している注文情報を parts_order 関数に設定し confirmed = false で実行して下さい。
* あなたの「最終確認です。以下の内容で部品を注文しますが、よろしいですか?」の問いかけに対して、
  ユーザーから肯定的な返答が確認できた場合のみ parts_order 関数を confirmed = true で実行し部品の注文を行って下さい。
* ユーザーから部品の注文の手続きをやめる、キャンセルする意思を伝えられた場合のみ、
  parts_order 関数を canceled = true で実行し、あなたはそれまでの部品の注文に関する内容を全て忘れます。

parts_order 関数を実行する際の注文する部品とその個数の扱い
----------------------------------------------------------
また、parts_order 関数を実行する際、注文する部品とその個数は part_no_and_quantities に設定して下さい。

part_no_and_quantities は注文する部品とその個数の表現する dict の list です。
list の要素となる各 dict は key として part_no と "quantity" を持ちます。
part_no の value が部品名称の文字列、 "quantity" の value が個数を意味する数字の文字列です。
以下は部品'J-26'を2個と部品'デカールC'を1枚注文する場合の part_no_and_quantities です。

```
[{"part_no": "J-26", "quantity": "2"}, {"part_no": "デカールC", "quantity": "1"}]
```

'''

# エージェントの初期化


class PartsOrderInput(BaseModel):
    name: str = Field(description="注文される方のお名前です。")
    kana: str = Field(description="注文される方のお名前（カタカナ）です。")
    post_code: str = Field(description="部品送付先の郵便番号です。")
    address: str = Field(description="部品送付先の住所です。")
    tel: str = Field(description="注文される方のお電話番号です。")
    email: str = Field(description="注文される方のメールアドレスです。")
    product_name: str = Field(
        description="部品注文の対象となる商品の名称です。例:'PG 1/24 ダンバイン'")
    product_no: str = Field(
        description="部品注文の対象となる商品の箱や説明書に記載されている6桁の数字の文字列です。")
    part_no_and_quantities: List[Dict[str, str]] = Field(description=(
        '注文する部品とその個数の表現する dict の list です。\n'
        'dict は key "part_no" の value が部品名称の文字列、key "quantity" の value が個数を意味する整数です。\n'
        '例: 部品"J-26"を2個と部品"デカールC"を1枚注文する場合は、\n'
        '\n'
        '[{"part_no": "J-26", "quantity": 2}, {"part_no": "デカールC", "quantity": 1}]\n'
        '\n'
        'としてください。'))
    confirmed: bool = Field(description=(
        "注文内容の最終確認状況です。最終確認が出来ている場合は True, そうでなければ False としてください。\n"
        "* confirmed が True の場合は部品の注文が行われます。 \n"
        "* confirmed が False の場合は注文内容の確認が行われます。")
    )
    canceled: bool = Field(description=(
        "注文の手続きを継続する意思を示します。\n"
        "通常は False としますがユーザーに注文の手続きを継続しない意図がある場合は True としてください。\n"
        "* canceled が False の場合は部品の注文手続きを継続します。 \n"
        "* canceled が True の場合は注文手続きをキャンセルします。")
    )


@tool("parts_order", return_direct=True, args_schema=PartsOrderInput)
def parts_order(
    name: str,
    kana: str,
    post_code: str,
    address: str,
    tel: str,
    email: str,
    product_name: str,
    product_no: str,
    part_no_and_quantities: List[Dict[str, str]],
    confirmed: bool,
    canceled: bool,
) -> str:
    """プラモデルの部品を紛失、破損した場合に必要な部品を個別注文します。注文の内容確認にも使用します"""
    if canceled:
        return "わかりました。また部品の注文が必要になったらご相談ください。"

    def check_params(name, kana, post_code, address, tel, email, product_name, product_no, part_no_and_quantities):
        for arg in [name, kana, post_code, address, tel, email, product_name, product_no]:
            if arg is None or arg == "***" or arg == "":
                return False
        if not part_no_and_quantities:
            return False

        # for p_and_q in part_no_and_quantities:
        #     part_no = p_and_q.get('part_no', '***')
        #     quantity = p_and_q.get('quantity', '***')
        #     if 'part_no' not in p_and_q or 'quantity' not in p_and_q:
        #         return False
        #     if part_no == '***' or quantity == '***':
        #         return False
        for p_and_q in part_no_and_quantities:
            if "part_no" not in p_and_q or "quantity" not in p_and_q:
                return False
            if p_and_q["part_no"] == "***" or p_and_q["quantity"] == "***":
                return False
        return True

    has_required = check_params(name, kana, post_code, address, tel, email, product_name, product_no, part_no_and_quantities)

    if part_no_and_quantities:
        part_no_and_quantities = "\n    ".join(
            [f"{row.get('part_no','***')} x {row.get('quantity','***')}"
             for row in part_no_and_quantities]
        )
    else:
        part_no_and_quantities = "***"
    # if part_no_and_quantities:
    #     parts_list = []
    #     for row in part_no_and_quantities:
    #         part_no = row.get('part_no', '***')
    #         quantity = row.get('quantity', '***')
    #         if part_no != '***' and quantity != '***':
    #             parts_list.append(f"{part_no} x {quantity}")
    #         else:
    #             parts_list.append("***")
    #     part_no_and_quantities = "\n    ".join(parts_list)
    # else:
    #     part_no_and_quantities = "***"

    # 注文情報のテンプレート
    order_template = (
        f'・お名前: {name}\n'
        f'・お名前(カナ): {kana}\n'
        f'・郵便番号: {post_code}\n'
        f'・住所: {address}\n'
        f'・電話番号: {tel}\n'
        f'・メールアドレス: {email}\n'
        f'・商品名: {product_name}\n'
        f'・商品番号: {product_no}\n'
        f'・ご注文の部品\n'
        f'    {part_no_and_quantities}'
    )

    # 追加情報要求のテンプレート
    request_information_template = (
        f'ご注文には以下の情報が必要です。"***" の項目を教えてください。\n'
        f'\n'
        f'{order_template}'
    )

    # 注文確認のテンプレート
    confirm_template = (
        f'最終確認です。以下の内容で部品を注文しますが、よろしいですか?\n'
        f'\n{order_template}'
    )

    # 注文完了のテンプレート
    complete_template = (
        f'以下の内容で部品を注文しました。\n'
        f'\n{order_template}\n'
        f'\n2営業日以内にご指定のメールアドレスに注文確認メールが届かない場合は、\n'
        f'弊社HPからお問い合わせください。'
    )

    if has_required and confirmed:
        # TODO invoke order here!
        return complete_template
    else:
        if has_required:
            return confirm_template
        else:
            return request_information_template


parts_order_tools = [parts_order]

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
chat_history = MessagesPlaceholder(variable_name='chat_history')

# モデルの初期化
llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
    model_kwargs={"top_p": 0.1, "function_call": {"name": "parts_order"}}
)

agent_kwargs = {
    "system_message": SystemMessagePromptTemplate.from_template(template=PARTS_ORDER_SYSTEM_PROMPT),
    "extra_prompt_messages": [chat_history]
}
parts_order_agent = initialize_agent(
    parts_order_tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=verbose,
    agent_kwargs=agent_kwargs,
    memory=memory
)


PARTS_ORDER_SSUFFIX_PROMPT = '''

重要な注意事項
--------------
注文情報に未知の項目がある場合は予測や仮定をせず "***" に置き換えてください。

parts_order 関数はユーザーから部品の注文の手続きをやめる、キャンセルする意思を伝えられた場合のみ canceled = true で実行して、
それまでの部品の注文に関する内容を全て忘れてください。。

parts_order 関数は次に示す例外を除いて confirmed = false で実行してください。

あなたの「最終確認です。以下の内容で部品を注文しますが、よろしいですか?」の問いかけに対して、
ユーザーから肯定的な返答が確認できた場合のみ parts_order 関数を confirmed = true で実行して部品を注文してください。

最終確認に対するユーザーの肯定的な返答なしで parts-order 関数を confirmed = true で実行することは誤発注であり事故になるので、固く禁止します。
'''

messages = []
messages.extend(parts_order_agent.agent.prompt.messages[:3])
messages.append(SystemMessagePromptTemplate.from_template(
    template=PARTS_ORDER_SSUFFIX_PROMPT),)
messages.append(parts_order_agent.agent.prompt.messages[3])
parts_order_agent.agent.prompt.messages = messages


# response = parts_order_agent.run("PG 1/24 ダンバインの部品を紛失したので注文したいです。")
# print(response)
while True:
    message = input(">> ")
    if message == "exit":
        break
    response = parts_order_agent.run(message)
    print(response)
