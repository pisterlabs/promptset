import openai
import sys
import os
from dotenv import load_dotenv
import glob


# OpenAI APIキーを設定
load_dotenv()
openai.api_key = os.environ["OPEN_API_KEY"]


def text_to_todo(text):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "TODOリストの下書きを作成してください。"
                                "TODOリストなのでtitle(タイトル)とPriority（重要度）とUrgency（緊急度）、deadline(期限)をセットにしてそれごとに複数書いてください。"
                                "出力は純粋な配列のJSON形式でお願いします。"
                                "{todo:["
                                    "{"
                                        "title:（タスクのタイトル50文字以内）\n,"
                                        "Priority:（A/B/C）\n,"
                                        "Urgency:（A/B/C）\n,"
                                        "deadline:（日付または時間が明確の場合。もし不明確ならブランク）," 
                                    "},"
                                "]}"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
        )

    # TODOリストを取得
    todo_list = response.choices[0].message.content.strip()
    return todo_list


def create_todo_list(file_path):
    # ファイルを開いて内容を読み込む
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # text_to_todo関数を使用してTODOリストを作成
    todo_list = text_to_todo(text)

    # file_pathから拡張子を除いたファイル名を取得し、_todo.txtを付加
    base_name = os.path.splitext(file_path)[0]
    todo_file_path = base_name + "_todo.txt"

    # 生成されたTODOリストを新しいファイルに書き出す
    with open(todo_file_path, 'w', encoding='utf-8') as file:
        file.write(todo_list)

    return todo_list


if __name__ == "__main__":
    # コマンドライン引数または標準入力からファイル名を取得
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("ファイル名を入力してください:").strip()

    for file in glob.glob(filename):
        print(file)
        todo = create_todo_list(file)

    input("\nなにかキーを押してください終了します:")
