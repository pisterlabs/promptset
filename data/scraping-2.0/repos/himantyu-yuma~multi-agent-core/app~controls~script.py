import datetime
import re

from bson import ObjectId
from langchain import PromptTemplate
from langchain.chains import LLMChain

from controls.llm import chatGpt
from repository.database.mongoDB import MongoDB
from settings.get_template import get_template


def create_script(
    topic: str, instruction: str, published_at: datetime.datetime
):
    """
    指定されたトピックについて、三姉妹の台本を作る
    Args:
        topic: str : トピック(DB保存用)
        instruction: str : 生成させる際の命令
    Returns:
        生成された台本のmongo上のID
    """
    template = get_template("CreateScript")

    prompt = PromptTemplate(
        template=template,
        input_variables=["topic"],
    )

    chain = LLMChain(llm=chatGpt, prompt=prompt)
    result = chain.run({"topic": instruction})

    lines = result.split("\n")
    scripts = []
    line_order = 0
    for line in lines:
        if line == "":
            continue
        (speaker, quote) = re.split("[:：]", re.sub("[ 　]", "", line))
        scripts.append(
            {"order": line_order, "speaker": speaker, "quote": quote}
        )
        line_order += 1

    data = {
        "topic": topic,
        "prompt": template.format(topic=instruction),
        "scripts": scripts,
        "published_at": published_at,
        "created_at": datetime.datetime.now(),
    }

    client = MongoDB(collection_name="scripts")
    scripts_id = client.insert_one(data).inserted_id
    client.close_connection()

    return str(scripts_id)


def get_script(script_id: str):
    # FIXME: 多分with句使えば良い感じになりそう
    client = MongoDB(collection_name="scripts")
    data = client.find_one({"_id": ObjectId(script_id)})
    result = {**data, "_id": str(data["_id"])}
    client.close_connection()
    return result


def filter_scripts(published_date: datetime.date | None):
    client = MongoDB(collection_name="scripts")
    if published_date is None:
        data = client.find_many({})
    else:
        data = client.find_many(
            {
                "published_at": {
                    # JSTでの日付を指定されたと仮定して変換
                    "$gte": datetime.datetime.combine(
                        published_date, datetime.time(0, 0, 0)
                    )
                    - datetime.timedelta(hours=9),
                    "$lt": datetime.datetime.combine(
                        published_date, datetime.time(0, 0, 0)
                    )
                    - datetime.timedelta(hours=9)
                    + datetime.timedelta(days=1),
                }
            }
        )
    result = [{**datum, "_id": str(datum["_id"])} for datum in data]
    return result


def update_script(script_id, data):
    client = MongoDB(collection_name="scripts")
    # data = {
    #     "topic": topic,
    #     "prompt": template.format(topic=instruction),
    #     "scripts": scripts,
    #     "published_at": published_at,
    #     "created_at": datetime.datetime.now(),
    # }
    client.update_one({"_id": ObjectId(script_id)}, {"$set": data})


if __name__ == "__main__":
    import pathlib
    from time import sleep

    # sample = create_script(
    #     "学食",
    #     "三姉妹は学食をこれから食べる予定であり、チキンタツタ丼がおすすめであることについて話している",
    #     datetime.datetime(2023, 5, 19, 12, 0, 0),
    # )
    # print(sample)
    current_dir = pathlib.Path(__file__).parent

    for i in range():
        sample = create_script(
            "雑談",
            "三姉妹が大学生活について雑談している。",
            datetime.datetime(2023, 6, 6, 13, i, 0),
        )
        with open(
            current_dir / "scripts.txt", mode="w", encoding="utf-8"
        ) as f:
            f.write(f"{sample}\n")
        print(sample)
        sleep(10)
