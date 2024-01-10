# storyID を入力すると各場面の本文の要約をファイルに保存する
import argparse
import os
import re
from openai import OpenAI
from chatgpt_utils import get_scene_num, GPT_KEY, GPT_MODEL
from summarize import summarize_one_scene
from multiprocessing import Pool

client = OpenAI(api_key=GPT_KEY)
gpt_model = GPT_MODEL


def wrapper(args):
    storyID, sceneID, show_log = args
    summary = summarize_one_scene(storyID, sceneID, show_log)
    with open(f"log/{storyID}/summary_scene{sceneID}.txt", "w", encoding="utf-8") as f:
        f.write(summary)


def main():
    # 入力で storyID を指定
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyID", type=int)
    parser.add_argument("--show_log", action="store_true")
    args = parser.parse_args()

    storyID = args.storyID
    show_log = args.show_log

    # すでに要約済みの場合, 実行しない
    if os.path.exists(f"log/{storyID}/summary.txt"):
        print("Summary has already existed!")
        exit()
    
    # 分割済みの本文が存在しない場合, 先に 1_preprocess_txt.py を実行するように促す
    if not os.path.exists(f"log/{storyID}/body_scene0.txt"):
        print("The splited text doesn't exist!")
        print(f"Please run 'python 1_preprocess_txt.py --storyID {storyID}")


    # 各場面の要約を作成し, summary.txt に保存する
    summaries = []
    scene_num = get_scene_num(storyID)
    # 各場面の要約を作成しファイルに保存する
    tasks = [(storyID, sceneID, show_log) for sceneID in range(scene_num)]
    with Pool(scene_num) as p:
        p.map(wrapper, tasks)
    
    # 各場面の要約を合わせて一つのファイルに保存する
    for sceneID in range(scene_num):
        with open(f"log/{storyID}/summary_scene{sceneID}.txt", encoding="utf-8") as f:
            summaries.append(f.read())
        os.remove(f"log/{storyID}/summary_scene{sceneID}.txt")

    with open(f"log/{storyID}/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summaries))


if __name__ == "__main__":
    main()
