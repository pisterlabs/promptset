import json
import os
import sqlite3
import openai

openai.api_key = "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"


def query_openai(headline: str, abstract: str):
    prompt = (
        "I have a news report which includes its headlines and abstract."
        + "\n\n"
        + "Headline: "
        + headline
        + "\n\n"
        + "Abstract: "
        + abstract
        + "\n\n"
        + "I need you to tell me something as below: \
        1. It's main topic (only one word or phrase) ，the topic can not be 'politics'! \
        2. It's report keywords (separate with english commas) \
        3. It's sentiment towards the main topic (only one word), choose one from [positive, negative, neutral] \
        4. The source bias of this media, choose one from [liberal, conservative, centrist], if you cannot tell the source bias, you have to guess one (only one word) \
        5. If one or more certain state in the United States is mentioned in the news, simply return the names of the state (only state name, if more than one, separate with commas!), else return None\
        \n\n \
        Do not explain, please place each answer on a new line, the answers are:"
        + "\n\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response = str(response)
    response = json.loads(response)
    result = response["choices"][0]["message"]["content"]
    answers = [item[2:].strip() for item in result.split("\n")]

    topic: str = answers[0]
    keywords: str = answers[1]
    sentiment: str = answers[2]
    bias: str = answers[3].lower()
    state: str = answers[4]

    if len(bias.split(" ")) > 1:
        bias = "centrist"

    if len(sentiment.split(" ")) > 1:
        sentiment = "neutral"

    state = 'None' if 'None' in state else state.strip()
    state = state[:-1] if state and state[-1] == '.' else state

    return {
        "topic": topic,
        "keywords": keywords,
        "sentiment": sentiment.lower(),
        "bias": bias.lower(),
        "state": state
    }


def analyze():
    conn = sqlite3.connect("./news.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM news")
    rows = cursor.fetchall()

    for row in rows:
        # 取出每条数据的各个字段
        id = row[0]
        headline = row[1]
        abstract = row[2]
        lead_paragraph = row[3]
        keywords = row[10]
        analyzed_topic = row[12]

        if analyzed_topic is not None:
            continue
        else:
            try:
                res = query_openai(
                    headline=headline, abstract=abstract or lead_paragraph
                )

                topic = res["topic"]
                keywords = res["keywords"]
                sentiment = res["sentiment"]
                bias = res["bias"]
                state = res["state"]
                print("id: " + str(id) + " is done!")

                # 更新数据库
                conn.execute(
                    """UPDATE news SET analyzed_topic = ?, analyzed_keywords = ?, analyzed_sentiment = ?, analyzed_bias = ?, state = ? WHERE id = ?""",
                    (topic, keywords, sentiment, bias, state, id),
                )
                conn.commit()
            except:
                # 如果出错了，就重新执行该脚本
                print("重新启动任务")
                os.system("python analyze.py")
                break

    conn.close()

if __name__ == '__main__':
    analyze()
