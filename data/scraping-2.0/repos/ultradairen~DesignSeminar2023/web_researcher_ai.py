import os, time
from dotenv import load_dotenv
import langchain
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper

import simpledcapi
from langchain.tools import tool
import logging
import re
import random

langchain.debug = True
load_dotenv()

# 孫悟空が行く国と言語
countries = [
    {"name": "日本", "lang": "日本語", "gl": "jp", "hl": "ja"},
    {"name": "アメリカ", "lang": "英語", "gl": "us", "hl": "en"},
    {"name": "中国", "lang": "中国語", "gl": "cn", "hl": "zh-cn"},
    {"name": "ドイツ", "lang": "ドイツ語", "gl": "de", "hl": "de"},
    {"name": "フランス", "lang": "フランス語", "gl": "fr", "hl": "fr"},
    {"name": "イタリア", "lang": "イタリア語", "gl": "it", "hl": "it"},
    {"name": "スペイン", "lang": "スペイン語", "gl": "es", "hl": "es"},
]

selected_country = {}

# 最大実行回数
max_execution_count = int(os.getenv("MAX_EXECUTION_COUNT", 1))

# 次回実行待ち時間の最小値（秒）
interval_sec_min = int(os.getenv("INTERVAL_SEC_MIN", 60))

# 次回実行待ち時間の最大値（秒）
interval_sec_max = int(os.getenv("INTERVAL_SEC_MAX", 3600))

# Discourseから持ってくる過去投稿の件数
latest_posts_count = int(os.getenv("DISCOURSE_LATEST_POSTS_COUNT", 10))


# Logging configuration
class UnicodeEscapeDecodeFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        message = message.replace("\\\\", "\\")
        message = message.replace("\\n", "\n")
        return re.sub(
            r"\\u([0-9a-fA-F]{4})", lambda x: chr(int(x.group(1), 16)), message
        )


logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
handler = logging.getLogger().handlers[0]
handler.setFormatter(UnicodeEscapeDecodeFormatter("%(levelname)s:%(name)s:%(message)s"))

# Discourse configuration
simpledcapi.discourse_url = os.getenv("DISCOURSE_URL")
simpledcapi.Api_Key = os.getenv("DISCOURSE_API_KEY")
simpledcapi.Api_Username = os.getenv("DISCOURSE_API_USERNAME")

# Target discourse category/topic
category_id = os.getenv("DISCOURSE_CATEGORY_ID")
topic_id = os.getenv("DISCOURSE_TOPIC_ID")

@tool
def search(query: str) -> str:
    """useful for when you need to answer questions about current events. You should ask targeted questions"""
    search = GoogleSerperAPIWrapper(
        gl=selected_country["gl"], hl=selected_country["hl"]
    )
    return search.results(query)["organic"][:10]


tools = [search]
model = os.getenv("OPENAI_MODEL")
llm = ChatOpenAI(model=model, temperature=1.0)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=25,
    early_stopping_method="generate",
)

question_template = f"""
あなたはアニメ ドラゴンボールの孫悟空で、仲間が以下の事業方針について検討していたことを知りました。以下の議論を参考に、議論に参考になる情報を{{language}}のキーワードを用いてインターネット検索し、みんなに教えてください。インターネットで検索したとは言わずに、例えば何かしらドラゴンボールのアニメの中で展開されるシーンの中にうまく入れ込んで情報を手に入れたことを言及してください。{{country}}を必ず会話に入れてください。ドラゴンボールの世界観にそって、孫悟空のような口調で喋ってください。敬語は禁止です。

Final Answerは日本語を利用し、インターネット検索に含まれていた関連URLも提示してください。マークダウンを利用してください。絵文字を必ず多用して答えてください。

# 事業方針
動物との共生を通じて、人々の幸福を追求する新規事業

# 今までの議論の履歴
{{latest_posts_formatted}}

# フォーマット
```
[孫悟空として一言]

## [タイトル](URL)
[議論に役立つと思ったポイントを短くコメント]

※3件程度繰り返し出力
```
"""

execution_count = 0

while execution_count < max_execution_count:
    logging.info("開始")

    # 孫悟空が行く国と言語をランダムに選択
    selected_country = random.choice(countries)

    # 孫悟空が行く国と言語をログに出
    logging.info(f"孫悟空が行く国：{selected_country['name']}")

    # 直近の投稿を取得
    latest_posts = simpledcapi.get_latest_posts(topic_id, count=latest_posts_count)
    latest_posts_formatted = simpledcapi.format_posts(latest_posts)

    # 最後の投稿のAgent名を取得
    latest_post = latest_posts[-1]
    latest_post_agent_name = latest_post["raw"].split("\n")[0] if latest_post else ""

    # 最後の投稿が自分かどうか確認
    if latest_post_agent_name == "孫悟空":
        logging.info("最後の投稿が自分。処理終了。")
    else:
        logging.info("最後の投稿が自分以外。処理開始。")

        query = question_template.format(
            language=selected_country["lang"],
            country=selected_country["name"],
            latest_posts_formatted=latest_posts_formatted,
        )
        message = agent.run(query)

        logging.info(f"Done. Message:\n{message}")

        # Discourseへ投稿
        # 返答先として最後の投稿の番号を取得
        post_number = latest_posts[-1]["post_number"]

        # 本文を作成
        body = f"孫悟空\n\n{message}"

        # Discourseへ投稿実施
        logging.info("Post to discourse.")
        simpledcapi.create_reply(body, topic_id, post_number)
        logging.info("Done.")

    execution_count += 1

    if execution_count < max_execution_count:
        random_interval = random.uniform(interval_sec_min, interval_sec_max)
        logging.info(f"sleep for {random_interval:.0f} sec")
        time.sleep(random_interval)
