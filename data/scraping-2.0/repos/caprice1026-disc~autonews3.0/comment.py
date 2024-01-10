from google.cloud import pubsub_v1
import os
import json
import random
import logging
from openai import OpenAI
import random
import base64
import requests
from requests.auth import HTTPBasicAuth



# 環境変数からAPIキーとその他の設定を取得
OPENAI_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI API呼び出し関数
def openai_api_call(model, temperature, messages, max_tokens, response_format):
    client = OpenAI(api_key=OPENAI_api_key)  # 非同期クライアントのインスタンス化
    try:
        # OpenAI API呼び出しを行う
        response = client.chat.completions.create(model=model, temperature=temperature, messages=messages, max_tokens=max_tokens, response_format=response_format)
        return response.choices[0].message.content  # 辞書型アクセスから属性アクセスへ変更
    except Exception as e:
        logging.error(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
        raise

# カテゴリーに応じたペルソナの配列
personas_by_category = {
    "blockchain-news": {
        "agree":{
            "Raj Patel": {"職業: ITコンサルタント, 性格: 知的、好奇心旺盛、実用的, 思想: テクノロジーの進歩を重視し、仮想通貨をビジネスの効率化ツールとして見ている, 宗教: ヒンドゥー教, 人種/民族: インド系イギリス人, バックグラウンド: ロンドンで育ち、情報技術で修士号を取得。大手企業からスタートアップまで、幅広いクライアントに対してデジタル変革を支援している。仮想通貨の技術的側面に強い関心を持つ。"},
            "Carlos Gutierrez":{" 職業: フィンテックスタートアップのCEO, 性格: 革新的、リスクテイカー、楽観的, 思想: 金融の民主化を信じ、仮想通貨を通じて銀行非対応者にも金融サービスを提供したい, 宗教: カトリック, 人種/民族: ヒスパニック系アメリカ人, バックグラウンド: マイアミで育ち、コンピュータサイエンスの学位を取得後、テクノロジーと金融の融合を推進する企業を立ち上げた。ブロックチェーンの可能性に情熱を注いでいる。"},
            "Emeka Okonkwo":{"職業: NGOのプロジェクトマネージャー, 性格: 献身的、協調性があり、思慮深い, 思想: 経済的包摂を推進し、途上国における仮想通貨の利用を支援, 宗教: キリスト教（プロテスタント派）, 人種/民族: ナイジェリア系, バックグラウンド: ナイジェリアのラゴスで育ち、国際開発学を学んだ後、地域コミュニティの発展に貢献する国際NGOで働いている。仮想通貨が金融アクセスを改善する手段としての可能性に注目している。"},
            "Maya Johnson":{"職業: ソーシャルメディアインフルエンサー, 性格: カリスマ的、創造的、社交的, 思想: デジタルネイティブ世代の代表として、仮想通貨のトレンドとライフスタイルへの統合を推進, 宗教: 無宗教, 人種/民族: アフリカ系カナダ人, バックグラウンド: トロントで育ち、マーケティングを学んだ後、フォロワー数百万人を抱えるソーシャルメディアアカウントを運営。仮想通貨をファッションやライフスタイルと結びつけるコンテンツを制作している。"},
            "Hiro Tanaka":{"職業: 投資家, 性格: 冒険的、決断力があり、自信家, 思想: 新たな投資機会を求め、仮想通貨市場のボラティリティを利用している, 宗教: 神道, 人種/民族: 日本人, バックグラウンド: 東京で金融を学び、国際的な投資ファンドで働いている。仮想通貨を投資の多様化と将来性のある資産と見做している。"}
        },
        "disagree":{
            "Nia Johnson":{"職業: 環境活動家, 性格: 熱心、共感的、決断力がある, 思想: 持続可能性と環境保護を重視し、仮想通貨のマイニングがもたらす環境問題に批判的, 宗教: プロテスタント, 人種/民族: アフリカ系アメリカ人, バックグラウンド: カリフォルニア州オークランドで生まれ、環境科学を学んだ後、気候変動に対する行動を強く訴えるNGOで働いている。仮想通貨のエネルギー消費に対して公然と批判している。"},
            "Elena Ivanova":{"職業: セキュリティアナリスト, 性格: 警戒心が強く、詳細にこだわり、信頼性が高い, 思想: デジタルセキュリティを重視し、仮想通貨のセキュリティリスクに対して警告を発している, 宗教: 無宗教, 人種/民族: ロシア系, バックグラウンド: モスクワ生まれでサイバーセキュリティに関する学位を持ち、多国籍企業でセキュリティ戦略を策定している。仮想通貨の安全性と規制の強化を主張している。"},
            "Lars Svensson":{"職業: システムエンジニア, 性格: 細部にこだわり、合理的、静か, 思想: 技術の進歩を重視し、仮想通貨の技術的側面やセキュリティの改善に注力, 宗教: ルーテル教会, 人種/民族: スウェーデン人, バックグラウンド: ストックホルムの工科大学でコンピュータサイエンスを学び、その後、テック企業でブロックチェーン技術の開発に携わる。仮想通貨の将来に対しては楽観的だが、技術的な課題には厳しい目を持っている。"},
            "Sarah Goldberg":{"職業: ジャーナリスト, 性格: 好奇心が強く、公平無私、徹底的, 思想: 情報の透明性を重視し、仮想通貨業界におけるニュースと動向を追及, 宗教: ユダヤ教, 人種/民族: アメリカ人（ユダヤ系）, バックグラウンド: ニューヨークでジャーナリズムを学び、主要なニュースメディアでテクノロジーと金融の分野を担当している。ブロックチェーン技術の社会的影響についての報道に力を入れている。"},
            "Zhang Wei":{"職業: 経済学者, 性格: 分析的、慎重、批判的, 思想: 仮想通貨の市場動向とその経済への影響を研究しており、規制の必要性を強調, 宗教: 仏教, 人種/民族: 中国系カナダ人, バックグラウンド: トロントで育ち、経済学で博士号を取得。現在は大学で教鞭を取りつつ、仮想通貨のリスクと経済に与える影響についての論文を数多く発表している。"}
        }
    },
    "ai-news": {
        "agree": {
            "Emma Lee": { "エマ・リー（32歳）はスタンフォード大学で工学博士号を取得し、革新的なAIソリューションに取り組むロボティクスエンジニアです。知的好奇心が旺盛で、合理的思考と技術的インサイトをもって、人間が直面する課題を解決するためにAIを応用しています。彼女は無宗教のアジア系アメリカ人として、テクノロジーと倫理の交点に関する議論にも深い見識を持ち、その分野でのジェンダー多様性に積極的に貢献しています。" },
            "Javier Garcia": { "ハビエル・ガルシア（45歳）は、実用主義的な観点から彼の中小企業を運営している経営者です。彼は野心的であり、リーダーシップとビジョンに溢れており、ビジネススクールで培った知見を活用して自身の企業を成功に導いています。カトリックの信仰を持つヒスパニックとして、社会への貢献と倫理的な経営を重視しながらも、AI技術を利用して業務の効率化と市場における競争力を高めています。" },
            "Olivia Janson": { "オリビア・ジャンソン（29歳）は、カリフォルニア州の公立学校で働く熱心な教育者です。彼女は生徒のポテンシャルを最大限に引き出すための教育的アプローチにAIの可能性を見出し、そのために共感的で開放的な性格を活かしています。プロテスタントの信仰に立脚しながらも、テクノロジーの進歩を教育に取り入れることによって生じるポジティブな変化に対してプログレッシブな視点を持っており、彼女のクラスでは常に最先端の教育ツールが使用され、学生たちが次世代のツールに対する適応力を身につけられるよう努めています。" },
            "Emilie Dubois": { "エミリー・デュボア（34歳）は、革新的な人工知能技術の研究に没頭するフランス出身の経験豊かなデータサイエンティストです。彼女は複雑なデータセットを精密に分析し、洞察を引き出すことに特に長けており、その才能をブリュッセルの急成長中のAIスタートアップ企業に奉仕しています。開かれた社会としてのヨーロッパの理念と育ての親であるフランスの啓蒙思想に根差した彼女は、AI技術が人間の能力を拡張し、より公平な世界を築くための重要なツールであるとの考えを持っています。エミリーは、テクノロジーの民主化を通じて社会の包摂性を高め、AIの可能性を全ての人に届けることを使命として日々の研究に邁進しています。"},
            "Anita Patel":{ "アニータ・パテル（38歳）は、プリンストン大学を卒業し、人命を救う放射線科の医師として尽力しています。慎重に物事を考える性格で、医療分野にAIを取り入れることにより、診断の精度を高め患者の生存率の向上を実現しています。ヒンドゥー教徒であり、南アジア系アメリカ人の彼女は、伝統的な価値観と最新の技術が融合する平衡点を模索しています。"}
        },
        "disagree": {
            "John Smith": { "ジョン・スミス（54歳）は長年地元コミュニティで働く工場監督です。彼は勤勉で現実主義者、新しいテクノロジーに取って代わられることへの懸念を持っています。AIと自動化により多くの同僚が仕事を失うのを目の当たりにしてきた彼は、技術の進歩がもたらす利益よりも人間の価値と雇用の安定性を重視しています。キリスト教徒で白人の彼は、地域社会の伝統的な価値観と労働倫理を守ることを優先し、AIによる変化を慎重に扱うべきだと考えています。" },
            "Susan Johnson": { "スーザン・ジョンソン（47歳）は、保守的な市民団体の活動家で、技術とプライバシーに関する問題に強い関心を持っています。デジタル技術に精通しており、AIが個人のデータをどのように利用するかについて懸念を抱いています。彼女はプライバシーを第一に考え、個人データの管理と透明性に重点を置く政策を支持しています。無宗教の白人として、データ保護と消費者の権利を擁護し、AIの監視文化への移行に警鐘を鳴らしています。" },
            "Alex Gonzalez": { "アレックス・ゴンザレス（35歳）は、ハイテク産業の重圧の中で生きる芸術家です。彼は創造性と人間の感情が技術には再現できないと信じており、AIが芸術や文化に及ぼす影響について懐疑的です。感受性が豊かで、伝統的な手法で作品を作ることに価値を見出しています。彼は仏教徒でヒスパニック系アメリカ人で、地域社会の芸術と手仕事の重要性を尊重し、テクノロジーに飲み込まれないように個人の表現力を守るために活動しています。" },
            "Emily Zhou": { "エミリー・チョウ（29歳）は、デジタルネイティブ世代の教育社会学者です。彼女はAIと教育の関係についての研究を行っており、テクノロジーが教育の平等性にどのように影響を及ぼすかに疑問を投げかけています。批判的思考を重視し、学生に対してもテクノロジーへの盲目的な依存から脱却するよう励ましています。多文化主義を信条とする彼女は、アジア系アメリカ人として、教育技術がもたらす潜在的な不平等を解消するために、教育の多様性とアクセスの改善を求めています。" },
            "Takashi Yamamoto": { "山本隆（50歳）は、日本の中堅企業で営業部長を務めています。彼は人と人との関係を大切にし、AIによって人間性が薄れることを危惧しています。長年の経験から、信頼関係と直接的なコミュニケーションを重視し、AIが対人関係にもたらす可能性のある冷淡さに懐疑的です。儒教の影響を受けた日本人として、社会的秩序と調和を尊重し、テクノロジーが倫理観や社会構造に与える影響に注意深く目を向けています。" }
        }
    },
    "最先端テクノロジー全般": {
        "agree": {
            "CET_Persona1": {},
            "CET_Persona2": {},
            "CET_Persona3": {},
            "CET_Persona4": {},
            "CET_Persona5": {}
        },
        "disagree": {
            "CET_Persona1": {},
            "CET_Persona2": {},
            "CET_Persona3": {},
            "CET_Persona4": {},
            "CET_Persona5": {}
        }
    }
}
# ペルソナ選択関数
def select_random_persona_by_category(category, position):
    category_personas = personas_by_category.get(category)

    if not category_personas:
        return None, None

    personas = category_personas.get(position, {})
    selected_persona = random.choice(list(personas.items())) if personas else (None, None)

    return selected_persona


def generate_opinion(content, category, position):
    try:
        persona_name, full_persona = select_random_persona_by_category(category, position)
        if persona_name:
            opinion = openai_api_call(
                "gpt-3.5-turbo-1106",
                0.6,
                [
                    {"role": "system", "content": f'あなたはペルソナである"{persona_name}"として意見を生成してください。この人は、"{full_persona}"提供された文章の内容に対し日本語で意見を生成してください。'},
                    {"role": "user", "content": content}
                ],
                500,
                {"type": "text"}
            )
            return f'{persona_name}: {opinion}'
    except Exception as e:
        logging.error(f"意見生成中にエラーが発生: {e}")
        return f"エラーが発生しました: {e}"

def post_comment_to_wordpress(post_id, author_name, content):
    try:
        # WordPressのURLとエンドポイント
        wordpress_url = os.environ['WORDPRESS_URL']
        url = f'{wordpress_url}/wp-json/wp/v2/comments'

        # ベーシック認証のためのユーザー名とパスワード
        # 認証情報を環境変数から取得
        username = os.environ['USERNAME']
        password = os.environ['PASSWORD']

        # ベーシック認証のためのヘッダーを作成
        credentials = username + ':' + password
        token = base64.b64encode(credentials.encode())
        headers = {'Authorization': 'Basic ' + token.decode('utf-8')}

        # WordPressへのコメント投稿
        response = requests.post(
            url,
            headers=headers,
            data={
                'post': post_id,
                'author_name': author_name,
                'content': content
            }
        )
        response.raise_for_status()

        return response.json()
        
    except requests.RequestException as e:
        logging.error(f"WordPressへのコメント投稿に失敗しました: {e}")
        return None

def main(event, context):
    try:
        message = base64.b64decode(event['data']).decode('utf-8')
        message = json.loads(message)
        logging.info(f"メッセージを受け取りました: {message}")

        content = message['content']
        category = message['category']
        post_id = message['post_id']

        # 賛成と反対の意見を生成し、それぞれをWordPressに投稿
        for position in ["agree", "disagree"]:
            opinion = generate_opinion(content, category, position)
            if opinion:
                author_name, comment_content = opinion.split(': ', 1)
                post_comment_to_wordpress(post_id, author_name, comment_content)
                
    except Exception as e:
        logging.error(f"メイン処理中にエラーが発生しました: {e}")