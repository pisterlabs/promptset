import openai
import sys
import json
import os
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), '.env.local')
load_dotenv(dotenv_path)

def get_template_elements(ad_type, context) -> str:

    openai.api_key = os.getenv('OPENAI_API_KEY')

    # ad_type = "comparison"
    ad_type = ad_type
    
    sample_context = """\
        商品名: シェイクパック
        商品カテゴリー: プロテイン
        商品の特徴:「わたしは、わたしらしく。」をコンセプトとした、“シェーカーなし”でおいしく飲むことができる個包装タイプのプロテイン『シェイクパック』。
        女性が1食に必要な33種類の栄養素がたっぷり入った大豆由来の植物性ウェルネスプロテインを、いつでもどこでも手軽に飲むことができます。
        """
    sample_prompt = create_advertisement_prompt(ad_type, sample_context)

    # contextは実際にはrequestから取得する
    # context_info = """\
    #     商品名: Full Cover Bikini
    #     商品カテゴリー: 男性用下着
    #     商品の特徴:素肌に直接触れる下着のために独自に開発した素材「nova wool® melty plus」を使用。汗蒸れ・汗冷え・汗臭を解消する消臭・抗菌機能に加え、素肌を清潔かつ快適に保つ調温・調湿機能に長けています。素肌へのストレスをクリアにし、第二の肌となってあなたの活動を支えます。食い込まないを追求した設計がストレスゼロな着用感を実現。
    #     """
    context_info = context.encode('unicode-escape')
    
    prompt = create_advertisement_prompt(ad_type, context_info)

    response_json_string = create_sample_json_string(ad_type)

    prompts = [
        {"role": "system", "content": """\
         あなたは自社のinstagram広告の作成担当者です。
         ユーザーから依頼された広告を作成してください。
         あなたはユーザーから、広告を作成するためのいくつかの情報受け取ります。
         その情報からinstagramの広告を作成して、その要素をJSONの形式で返却してください。
         """},
        {"role": "user", "content": sample_prompt},
        {"role": "assistant", "content": response_json_string},
        {"role": "user", "content": prompt}
    ]

    

    try:
        res = openai.ChatCompletion.create(
            messages=prompts,#.encode('unicode-escape'), なぜなのなんなの
            model="gpt-3.5-turbo"
        )

        # APIからの戻り値のチェック
        if not res.get("choices"):
            raise ValueError("Unexpected response format from OpenAI API.")
    except Exception as e:
        print(f"Error occurred while making API call: {e}", file=sys.stderr)
        return "error"

    # return json.loads(res["choices"][0]["message"]["content"])
    return json.loads(res["choices"][0]["message"]["content"])


def create_advertisement_prompt(ad_type, context_info):
    """
    指定された広告タイプに基づいて、広告メッセージを作成します。

    :param ad_type: 広告のタイプ（例：'comparison', 'features', 'sale'など）。
    :param context_info: 広告内容を作成するためのコンテキスト情報。
    :return: 広告メッセージのリスト。
    """
    # 広告のタイプに基づいてメッセージのテンプレートを選択
    if ad_type == "comparison":
        # 比較広告のテンプレート
        template = """\
        以下の情報を参考にして、自社商品と他社商品の比較広告を作成してください。
        広告には以下の項目を含みます。
        参考情報には他社製品の特徴は含まれないので、自社製品に対して劣っていることを表現する文言を適当に作成してください。
        作成した内容を返却値例の形でJSONで返却してください。

        ・自社製品名
        ・自社製品の特徴1
        ・自社製品の特徴2
        ・自社製品の特徴3
        ・他社製品名
        ・（自社製品の特徴1に対応する）他社製品の特徴1
        ・（自社製品の特徴2に対応する）他社製品の特徴2
        ・（自社製品の特徴2に対応する）他社製品の特徴2

        返却値例:{sample_sale_json}
        参考情報:{context_info}
        """  
    elif ad_type == "features":
        # 特徴広告のテンプレート
        template = """\
        以下の情報を参考にして、自社の商品を紹介する広告を作成してください。
        広告には以下の項目を含みます。

        ・商品名
        ・商品の特徴1
        ・商品の特徴2
        ・商品の特徴3
        ・商品の特徴4
        ・商品の特徴5
        ・商品の特徴1～5の要約

        返却値例:{sample_sale_json}
        参考情報:{context}
        """  
    elif ad_type == "sale":
        # セール広告のテンプレート
        template = """\
        以下の情報を参考にして、セール広告を作成してください。
        セール広告には以下の項目を含みます。季節感や時期を考慮して作成してください。
        作成した内容を返却値例の形でJSONで返却してください。返却値例以外のkeyは含まないでください。

        ・メインのタイトル
        ・メインのメッセージ
        ・セール期間

        返却値例:{sample_sale_json}
        参考情報:{context_info}
        """  
    else:
        return ""  # 不明な広告タイプの場合は空のリストを返す

    # テンプレートにコンテキスト情報を注入
    prompt = template.format(sample_sale_json=create_sample_json_string(ad_type), context_info=context_info)

    # 作成されたメッセージを含むリストを返す
    return prompt

def create_sample_json_string(ad_type):
    """
    指定された広告タイプに基づいて、広告内容のJSONを作成します。

    :param ad_type: 広告のタイプ（例：'comparison', 'features', 'sale'など）。
    :return: 広告内容のJSON。
    """

    # 広告のタイプに基づいてJSONの内容を動的に変更
    if ad_type == "comparison":
        # 比較広告用のフィールドを追加/更新
        sample_json = """
            {
                "our_product": {
                    "name": "シェイクパック",
                    "features": [
                    "シェーカーなしで飲める",
                    "個包装タイプだから持ち運びも簡単",
                    "女性が1食に必要な33種類の栄養素がたっぷり"
                    ]
                },
                "competitor_product": {
                    "name": "プロテインA",
                    "features": [
                    "シェーカーが必要",
                    "大袋で持ち運びが難しい",
                    "男性向け"
                    ]
                }
            }
        """
    elif ad_type == "features":
        sample_json = """
            {
            "product_info": {
                "name": "シェイクパック",
                "features": [
                "シェーカーなしで飲める",
                "個包装タイプだから持ち運びも簡単",
                "女性が1食に必要な33種類の栄養素がたっぷり",
                "大豆由来の植物性ウェルネスプロテイン",
                "砂糖と人工甘味料は不使用"
                ],
                "features_summary": "“シェーカーなし”でおいしく飲むことができる個包装タイプのプロテイン『シェイクパック』。女性が1食に必要な33種類の栄養素がたっぷり入った大豆由来の植物性ウェルネスプロテインを、いつでもどこでも手軽に飲むことができます。"
            }
}
        """
    elif ad_type == "sale":
        sample_json = """
            {
            "campaign_info": {
                "main_titile": "冬のフラッシュセール2023",
                "main_message": "超ホットな最新アイテム",
                "sale_period": "2023-12-24～2023-12-31"
            }
            }
        """
    else:
        # 不明な広告タイプの場合はエラーを表示または空のJSONを返す
        return ""

    return sample_json

def create_advertisement_json(ad_type, product_info):
    """
    指定された広告タイプに基づいて、広告内容のJSONを作成します。

    :param ad_type: 広告のタイプ（例：'comparison', 'features', 'sale'など）。
    :param product_info: 商品に関する情報や広告内容に必要なその他の情報。
    :return: 広告内容のJSON。
    """
    # 基本的なJSON構造を定義
    ad_json_structure = {
        "product": {
            "name": product_info.get("name"),
            # その他の共通フィールド...
        },
        # 広告タイプに特有のフィールド...
    }

    # 広告のタイプに基づいてJSONの内容を動的に変更
    if ad_type == "comparison":
        # 比較広告用のフィールドを追加/更新
        ad_json_structure.update({
            "comparison": {
                # 比較に必要なフィールド...
            }
        })
    elif ad_type == "features":
        # 特徴広告用のフィールドを追加/更新
        ad_json_structure.update({
            "features": {
                # 特徴に関するフィールド...
            }
        })
    elif ad_type == "sale":
        # セール広告用のフィールドを追加/更新
        ad_json_structure.update({
            "sale": {
                # セール情報に関するフィールド...
            }
        })
    else:
        # 不明な広告タイプの場合はエラーを表示または空のJSONを返す
        return {}

    return ad_json_structure


if __name__ == "__main__":
    main()
