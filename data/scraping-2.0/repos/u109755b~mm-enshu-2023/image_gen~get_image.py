# 2023/11/24 菊地 ver.0作成

import os
import openai
import csv
import requests
from openai import OpenAI
from datetime import datetime

def load_api_key(file_path):
    """ファイルからAPIキーを読み込む"""
    with open(file_path, 'r') as file:
        return file.read().strip()

def send_dalle_request(api_key, prompt, image_quality):
    """DALL-E APIにリクエストを送信し、レスポンスを取得する"""
    quality_mapping = {'high': '1024x1024', 'middle': '512x512', 'low': '256x256'}
    image_quality = quality_mapping.get(image_quality, '512x512')  # デフォルトは'middle'
    
    openai.api_key = api_key
    response = openai.Image.create(prompt=prompt, n=1, size=image_quality)
    return response

def save_image(image_url, save_path):
    """画像データを指定されたパスに保存する"""
    # 保存先ディレクトリの作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 画像データの取得
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as image_file:
            image_file.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download image: {response.status_code}")


def log_action(log_file_path, log_entry):
    """操作をCSVファイルにログとして記録する"""
    with open(log_file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(log_entry)

def fetch_and_save_dalle_image_with_logging(novel_name, character_name, api_key_file_path, image_quality, log_file_path):
    """メイン関数：DALL-E APIを使用して画像を取得し、保存し、ログを記録する"""
    # api_key = load_api_key(api_key_file_path)
    client = OpenAI(api_key_file_path)
    prompt = f"{novel_name}: {character_name}"
    response = client.images.generate(
        model='dall-e-3',
        prompt=prompt,
        size='1024x1024',
        quality='standard',
        n=1
    )

    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'novel_name': novel_name,
        'character_name': character_name,
        'image_quality': image_quality,
        'status': 'Success' if response is not None else 'Failure'
    }

    if response is not None:
        image_id = response['data'][0]['id']
        save_path = save_image(image_id, novel_name, character_name)
        print(f"画像は {save_path} に保存されました")
    else:
        print("DALL-E APIから画像を取得できませんでした")

    log_action(log_file_path, log_entry)

# 使用例
# fetch_and_save_dalle_image_with_logging("Example Novel", "Example Character", "path/to/api_key.txt", "1024x1024", "path/to/log.csv")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fetch and save an image from DALL-E API.')
    parser.add_argument('--novel_name', type=str, help='The name of the novel.', default='My Novel')
    parser.add_argument('--character_name', type=str, help='The name of the character.', default='Main Character')
    parser.add_argument('--api_key_file_path', type=str, help='The path to the API key file.', default='./api_key.txt')
    parser.add_argument('--image_quality', type=str, choices=['high', 'middle', 'low'], help='The desired image quality.', default='middle')
    parser.add_argument('--log_file_path', type=str, help='The path to the log file.', default='./log.csv')

    args = parser.parse_args()

    fetch_and_save_dalle_image_with_logging(args.novel_name, args.character_name, args.api_key_file_path, args.image_quality, args.log_file_path)
