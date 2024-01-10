from flask import Flask, request, jsonify
import openai
import json

# OpenAI APIの認証情報をjsonファイルから読み込む
with open("../config.json", "r") as f:
    config = json.load(f)
    openai.api_key = config["OPENAI_API_KEY"]

def generate_advice(data):
    prompt_text = f"以下の属性を持つ人がダイエットしたいと考えています。この人がダイエット目標を達成するために、今日食べた食事や運動量の情報から、ダイエットアドバイザーのプロフェッショナルとして具体的なアドバイスを出力して\n\
    出力に関する制約条件：\n\
    ・{data['character_traits']}な{data['character_type']}のキャラクターの口調を強調\n\
    ・アドバイス以外の文面は出力しない\n\
    ・絵文字を多用\n\
    ・3文以内\n\
    ・100文字以内\n\
    ・最初に今日の摂取カロリーの上限に関する言及をすること\n\
    ・すでに食べた食事がある場合、バランスを考えて次に食べるべき食事の種類をお勧めすること\n\
    ・摂取カロリーが今日の摂取カロリーの上限を超えている場合は、摂取カロリーに関する言及をすること\n\
    ユーザーの属性：\n\
    ・現在の体重：{data['weight']}kg\n\
    ・身長：{data['height']}cm\n\
    ・年齢：{data['age']}歳\n\
    ・性別：{data['gender']}\n\
    今日の記録\n\
    ・摂取カロリーの上限：{data['target_calories']}kcal\n\
    ・消費カロリー：{data['calories_burned']}kcal\n\
    ・食事：{data['food']}\n\
    ・摂取カロリー：{data['calories_ate']}kcal\n\
    目標\n\
    ・目標体重：{data['target_weight']}㎏\n\
    ・目標期間：{data['target_period']}日\n\
    アドバイス："

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.8,
        max_tokens=200
    )
    advice = response.choices[0].message["content"]

    return advice