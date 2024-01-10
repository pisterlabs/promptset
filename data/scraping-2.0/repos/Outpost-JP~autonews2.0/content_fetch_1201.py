import functions_framework
import threading
import flask
from markupsafe import escape
import requests
import json
import os
from backoff import expo, on_exception
from bs4 import BeautifulSoup
import traceback
import langchain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64  # base64の重複インポートを削除
import logging  # loggingの重複インポートを削除
from openai import OpenAI
import gspread

def summarize_content(content):
    try:
        # テキストを分割するためのスプリッターを設定
        text_splitter = CharacterTextSplitter(
            chunk_size=5000,  # 分割するチャンクのサイズ
            chunk_overlap=100,  # チャンク間のオーバーラップ
            separator="\n"    # 文章を分割するためのセパレータ
        )
        texts = text_splitter.create_documents([content])

        # 要約チェーンを実行
        result = refine_chain({"input_documents": texts}, return_only_outputs=True)

        # 要約されたテキストを結合して返す
        return result["output_text"]
    except Exception as e:
        logging.error(f"要約処理中にエラーが発生しました: {e}")
        traceback.print_exc()
        return None

# 定数
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')  
GOOGLE_CREDENTIALS_BASE64 = os.getenv('CREDENTIALS_BASE64')   
OPENAI_api_key = os.getenv('OPENAI_API_KEY')
EXCLUDED_DOMAINS = ['github.com', 'youtube.com', 'wikipedia.org', 'twitter.com', 'www.youtube.com']

# プロンプトテンプレートの定義
refine_first_template = """以下の文章は、長い記事をチャンクで分割したものの冒頭の文章です。それを留意し、次の文章の内容と結合することを留意したうえで以下の文章をテーマ毎にまとめて下さい。
------
{text}
------
"""

refine_template = """下記の文章は、長い記事をチャンクで分割したものの一部です。また、「{existing_answer}」の内容はこれまでの内容の要約である。そして、「{text}」はそれらに続く文章です。それを留意し、次の文章の内容と結合することを留意したうえで以下の文章をテーマ毎にまとめて下さい。できる限り多くの情報を残しながら日本語で要約して出力してください。
------
{existing_answer}
{text}
------
"""
refine_first_prompt = PromptTemplate(input_variables=["text"],template=refine_first_template)
refine_prompt = PromptTemplate(input_variables=["existing_answer", "text"],template=refine_template)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
# 要約チェーンの初期化
refine_chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=refine_first_prompt,
    refine_prompt=refine_prompt
)

# gspread初期化
def init_gspread():

    # Base64デコード
    creds_json = base64.b64decode(GOOGLE_CREDENTIALS_BASE64).decode('utf-8')

    # JSONパース  
    creds = json.loads(creds_json)

    # gspread認証
    gc = gspread.service_account_from_dict(creds)  
    gc.session.timeout = 300

    # スプレッドシートオープン
    spreadsheet = gc.open_by_key(SPREADSHEET_ID)

    # 2枚目のシート取得
    worksheet = spreadsheet.get_worksheet(3)
    return worksheet


SHEET_CLIENT = init_gspread()



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

# URLからコンテンツを取得する関数
def fetch_content_from_url(url):
    try:
        logging.info(f"URLからコンテンツの取得を開始: {url}")

        # ユーザーエージェントを設定
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        response = requests.get(url, headers=headers, timeout=100)
        content = response.text

        logging.info(f"URLからコンテンツの取得が成功: {url}")
        return content

    except Exception as e:
        logging.warning(f"URLからのコンテンツ取得中にエラーが発生しました: {e}")
        return None

#　コンテンツをパースする関数 
def parse_content(content):
    try:
        # HTMLコンテンツをBeautiful Soupでパース
        soup = BeautifulSoup(content, 'html.parser')

        # ヘッダーとフッターを削除（もし存在する場合）
        header = soup.find('header')
        if header:
            header.decompose()

        footer = soup.find('footer')
        if footer:
            footer.decompose()

        # JavaScriptとCSSを削除
        for script in soup(["script", "style"]):
            script.decompose()

        # HTMLタグを削除してテキストのみを取得
        text = soup.get_text()

        # 改行をスペースに置き換え
        parsed_text = ' '.join(text.split())

        # パースされたテキストの文字数を出力
        print(f"パースされたテキストの文字数: {len(parsed_text)}")

        return parsed_text

    except Exception as e:
        logging.warning(f"コンテンツのパース中にエラーが発生しました: {e}")
        return None

# ランダムペルソナを選択する関数
def select_random_persona():
    # 定義されたペルソナの辞書
    personas = {
        1: "Raj Patel - 職業: ITコンサルタント, 性格: 知的、好奇心旺盛、実用的, 思想: テクノロジーの進歩を重視し、仮想通貨をビジネスの効率化ツールとして見ている, 宗教: ヒンドゥー教, 人種/民族: インド系イギリス人, バックグラウンド: ロンドンで育ち、情報技術で修士号を取得。大手企業からスタートアップまで、幅広いクライアントに対してデジタル変革を支援している。仮想通貨の技術的側面に強い関心を持つ。",
        2: "Nia Johnson - 職業: 環境活動家, 性格: 熱心、共感的、決断力がある, 思想: 持続可能性と環境保護を重視し、仮想通貨のマイニングがもたらす環境問題に批判的, 宗教: プロテスタント, 人種/民族: アフリカ系アメリカ人, バックグラウンド: カリフォルニア州オークランドで生まれ、環境科学を学んだ後、気候変動に対する行動を強く訴えるNGOで働いている。仮想通貨のエネルギー消費に対して公然と批判している。",
        3: "Zhang Wei - 職業: 経済学者, 性格: 分析的、慎重、批判的, 思想: 仮想通貨の市場動向とその経済への影響を研究しており、規制の必要性を強調, 宗教: 仏教, 人種/民族: 中国系カナダ人, バックグラウンド: トロントで育ち、経済学で博士号を取得。現在は大学で教鞭を取りつつ、仮想通貨のリスクと経済に与える影響についての論文を数多く発表している。",
        4: "Carlos Gutierrez - 職業: フィンテックスタートアップのCEO, 性格: 革新的、リスクテイカー、楽観的, 思想: 金融の民主化を信じ、仮想通貨を通じて銀行非対応者にも金融サービスを提供したい, 宗教: カトリック, 人種/民族: ヒスパニック系アメリカ人, バックグラウンド: マイアミで育ち、コンピュータサイエンスの学位を取得後、テクノロジーと金融の融合を推進する企業を立ち上げた。ブロックチェーンの可能性に情熱を注いでいる。",
        5: "Sarah Goldberg - 職業: ジャーナリスト, 性格: 好奇心が強く、公平無私、徹底的, 思想: 情報の透明性を重視し、仮想通貨業界におけるニュースと動向を追及, 宗教: ユダヤ教, 人種/民族: アメリカ人（ユダヤ系）, バックグラウンド: ニューヨークでジャーナリズムを学び、主要なニュースメディアでテクノロジーと金融の分野を担当している。ブロックチェーン技術の社会的影響についての報道に力を入れている。",
        6: "Hiro Tanaka - 職業: 投資家, 性格: 冒険的、決断力があり、自信家, 思想: 新たな投資機会を求め、仮想通貨市場のボラティリティを利用している, 宗教: 神道, 人種/民族: 日本人, バックグラウンド: 東京で金融を学び、国際的な投資ファンドで働いている。仮想通貨を投資の多様化と将来性のある資産と見做している。",
        7: "Elena Ivanova - 職業: セキュリティアナリスト, 性格: 警戒心が強く、詳細にこだわり、信頼性が高い, 思想: デジタルセキュリティを重視し、仮想通貨のセキュリティリスクに対して警告を発している, 宗教: 無宗教, 人種/民族: ロシア系, バックグラウンド: モスクワ生まれでサイバーセキュリティに関する学位を持ち、多国籍企業でセキュリティ戦略を策定している。仮想通貨の安全性と規制の強化を主張している。",
        8: "Emeka Okonkwo - 職業: NGOのプロジェクトマネージャー, 性格: 献身的、協調性があり、思慮深い, 思想: 経済的包摂を推進し、途上国における仮想通貨の利用を支援, 宗教: キリスト教（プロテスタント派）, 人種/民族: ナイジェリア系, バックグラウンド: ナイジェリアのラゴスで育ち、国際開発学を学んだ後、地域コミュニティの発展に貢献する国際NGOで働いている。仮想通貨が金融アクセスを改善する手段としての可能性に注目している。",
        9: " Maya Johnson - 職業: ソーシャルメディアインフルエンサー, 性格: カリスマ的、創造的、社交的, 思想: デジタルネイティブ世代の代表として、仮想通貨のトレンドとライフスタイルへの統合を推進, 宗教: 無宗教, 人種/民族: アフリカ系カナダ人, バックグラウンド: トロントで育ち、マーケティングを学んだ後、フォロワー数百万人を抱えるソーシャルメディアアカウントを運営。仮想通貨をファッションやライフスタイルと結びつけるコンテンツを制作している。",
        10: "Lars Svensson - 職業: システムエンジニア, 性格: 細部にこだわり、合理的、静か, 思想: 技術の進歩を重視し、仮想通貨の技術的側面やセキュリティの改善に注力, 宗教: ルーテル教会, 人種/民族: スウェーデン人, バックグラウンド: ストックホルムの工科大学でコンピュータサイエンスを学び、その後、テック企業でブロックチェーン技術の開発に携わる。仮想通貨の将来に対しては楽観的だが、技術的な課題には厳しい目を持っている。"
    }

    # 1から5までのランダムな整数を生成
    random_number = random.randint(1, 10)

    # 生成された整数に対応するペルソナを選択
    selected_persona = personas[random_number]
    # ペルソナの名前を抽出
    persona_name = selected_persona.split(" - ")[0]
    return selected_persona, persona_name

# 意見を生成する関数 (統合版)
def generate_opinion(content):
    try:
        full_persona, persona_name = select_random_persona()
        opinion = openai_api_call(
            "gpt-3.5-turbo-1106",
            0.6,
            [
                {"role": "system", "content": f'あなたは"""{full_persona}"""です。提供された文章の内容に対し日本語で意見を生成してください。'},
                {"role": "user", "content": content}
            ],
            2000,
            {"type": "text"}
        )
        opinion_with_name = f'{persona_name}: {opinion}'
        return opinion_with_name
    except Exception as e:
        logging.error(f"意見生成中にエラーが発生: {e}")
        return f"エラーが発生しました: {e}"

    # スプレッドシートに書き出す



    # スプレッドシートに書き出す
@on_exception(expo, gspread.exceptions.APIError, max_tries=3)
@on_exception(expo, gspread.exceptions.GSpreadException, max_tries=3)
def write_to_spreadsheet(row):
    if not SHEET_CLIENT:
        logging.error("スプレッドシートのクライアントが初期化されていません。")
        return False
    try:
        logging.info(f"スプレッドシートへの書き込みを開始: {row}")
        # スプレッドシートの初期化
        worksheet = SHEET_CLIENT

        # スプレッドシートに指定行に挿入
        worksheet.insert_row(row, 2)  # A2からD2に行を挿入

        logging.info(f"スプレッドシートへの書き込みが成功: {row}")

    except gspread.exceptions.APIError as e:
        logging.warning(f"一時的なエラー、リトライ可能: {e}")
        raise 

    except gspread.exceptions.GSpreadException as e:
        logging.error(f"致命的なエラー: {e}")
        raise
# メインのタスクの部分
def heavy_task(article_title, article_url):
    try:
        # URLからコンテンツを取得し、パースする
        content = fetch_content_from_url(article_url)
        if content is None:
            logging.warning(f"コンテンツが見つからない: {article_url}")
            return

        parsed_content = parse_content(content)
        if parsed_content is None:
            logging.warning(f"コンテンツのパースに失敗: {article_url}")
            return

        # parsed_contentが10000文字以下なら直接OpenAIに渡す
        if len(parsed_content) <= 10000:
            final_summary = openai_api_call(
                "gpt-4-1106-preview",
                0,
                [
                    {"role": "system", "content": "あなたは優秀な要約アシスタントです。提供された文章の内容を出来る限り残しつつ、日本語で要約してください。"},
                    {"role": "user", "content": parsed_content}
                ],
                4000,
                {"type": "text"}
            )
            if not final_summary:
                logging.warning(f"要約の洗練に失敗: {article_url}")
                return None
        else:
            # 初期要約を生成
            preliminary_summary = summarize_content(parsed_content)
            if preliminary_summary is None:
                logging.warning(f"コンテンツの要約に失敗: {article_url}")
                return

            # OpenAIを使用してさらに要約を洗練
            final_summary = openai_api_call(
                "gpt-4-1106-preview",
                0,
                [
                    {"role": "system", "content": "あなたは優秀な要約アシスタントです。提供された文章の内容を出来る限り残しつつ、日本語で要約してください。テーマごとに分割してリスト形式にすることは行わないでください。"},
                    {"role": "user", "content": preliminary_summary}
                ],
                4000,
                {"type": "text"}
            )

            if not final_summary:
                logging.warning(f"要約の洗練に失敗: {article_url}")
            return None
        
        # リード文生成のためのOpenAI API呼び出し
        try:
            lead_sentence = openai_api_call(
            "gpt-4",
            0,
            [
                {"role": "system", "content": "あなたは優秀なライターです。この要約のリード文（導入部）を簡潔に1～2センテンス程度でで作成してください。"},
                {"role": "user", "content": final_summary}
            ],
            150,  # リード文の最大トークン数を適宜設定
            {"type": "text"}
            )
            if not lead_sentence:
                logging.warning(f"リード文の生成に失敗: {article_url}")
                lead_sentence = "リード文の生成に失敗しました。"
        except Exception as e:
            logging.error(f"リード文生成中にエラーが発生: {e}")
            lead_sentence = "リード文の生成中にエラーが発生しました。"
        
        
        # ThreadPoolExecutorを使用して意見を並列生成
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_opinion, final_summary) for _ in range(3)]

            opinions = []
            for future in as_completed(futures):
                result = future.result()
                if result.startswith("エラーが発生しました"):
                    logging.warning(f"意見生成中にエラーが発生: {result}")
                else:
                    opinions.append(result)

        if not opinions:
            logging.warning(f"すべての意見生成関数がエラーをスローしました: {article_url}")

        
        # スプレッドシートに書き込む準備
        spreadsheet_content = [article_title, article_url, final_summary, lead_sentence] + opinions

        # スプレッドシートに書き込む
        write_to_spreadsheet(spreadsheet_content)
        logging.info(f"処理完了: {article_url}")

    except Exception as e:
        logging.error(f"{article_url} の処理中にエラーが発生: {e}")
        traceback.print_exc()

@functions_framework.http
def process_inoreader_update(request):
    request_json = request.get_json()

    if request_json and 'items' in request_json:
        for item in request_json['items']:
            article_title = escape(item.get('title', ''))
            article_href = escape(item['canonical'][0]['href']) if 'canonical' in item and item['canonical'] else ''


            # news.google.comを含むURLをスキップする
            if 'news.google.com' in article_href:
                logging.info(f"news.google.comのURLはスキップされます: {article_href}")
                continue

            if article_title and article_href:
                # 重い処理を非同期で実行するために別のスレッドを起動
                thread = threading.Thread(target=heavy_task, args=(article_title, article_href))
                thread.start()
        # メインスレッドでは即座に応答を返す
        return '記事の更新を受け取りました', 200
    else:
        return '適切なデータがリクエストに含まれていません', 400
         

