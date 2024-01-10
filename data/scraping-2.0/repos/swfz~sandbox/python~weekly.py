import openai
import sys
import datetime
import os


start_date = sys.argv[1]
end_date = sys.argv[2]

def date_range(start_date, end_date):
    delta = datetime.timedelta(days=1)
    while start_date <= end_date:
        yield start_date
        start_date += delta

def read_daily_note(date):
    file_path = f"/mnt/c/Users/sawafuji_yuya/src/obsidian/scrawl/daily_note/{date}.md"
    with open(file_path, 'r', encoding="utf-8") as f:
        return f.read()

def concat_files(start_date, end_date, folder_path):
    files_to_concat = [os.path.join(folder_path, date.strftime('%Y-%m-%d') + '.md') for date in date_range(start_date, end_date)]
    existing_files = list(filter(os.path.exists, files_to_concat))
    contents = list(map(read_file, existing_files))

    return "".join(contents)

def read_file(file):
    with open(file, "r") as f:
        return f.read()


def summary_of_day(date):
    system_content = """
あなたは私の秘書です
特定の日のDailyNoteを与えるのでその日がどんな日だったか要約してください
仕事、Private両方コンテンツがあると嬉しいです
DailyNodeでは次のようなことを書いています

## W
やったこと

## W
わかったこと

## T
次やること

## ToDo
ToDoリスト、`- [ ] タスク`は未実施のタスク、`- [x] タスク`は実施済みタスク

最後に、あなたが思う私についてのフィードバックを返答してください
"""

    print("------------------------------------------------------------")
    print(date.strftime("%Y-%m-%d") + ": 要約中.....")

    daily_note_content = read_daily_note(date)
    res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": daily_note_content},
            ]
        )

    print(res.choices[0]["message"]["content"].strip())


    return res.choices[0]["message"]["content"]


def summary(start_date, end_date):
    system_content = """
あなたは私の秘書です
一定期間の間の要約を与えるのでその期間どんなことがあったか、どういう期間だったかを要約してください
各日のフィードバックも表示してください
"""

    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    daily_note_summaries = list(map(summary_of_day, date_range(start_date, end_date)))
    contents = "".join(daily_note_summaries)

    # print("結合されたファイルの内容:")
    # print(daily_note_contents)
    print(f"{start_date} ~ {end_date} の期間を要約中.....")

    res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": contents},
            ]
        )

    print(res.choices[0]["message"]["content"].strip())


summary(start_date, end_date)

