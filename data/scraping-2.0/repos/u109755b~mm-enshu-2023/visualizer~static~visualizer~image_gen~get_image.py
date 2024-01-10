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

def fetch_and_save_dalle_image_with_logging(novel_name, character_name, character_description,api_key_file_path, image_save_path,log_file_path):
    """メイン関数：DALL-E APIを使用して画像を取得し、保存し、ログを記録する"""
    api_key = load_api_key(api_key_file_path)
    openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    prompt = (
    f"Create a digital art style image of a character from the novel '{novel_name}'. "
    f"The character, named '{character_name}', is described as follows: {character_description}. "
    "The image should focus solely on this character in a digital art style, without any other characters from the novel present."
    )

    
    image_quality = '1024x1024'
    
    response = client.images.generate(
        model='dall-e-3',
        prompt=prompt,
        size=image_quality,
        quality='standard',
        n=1
    )

    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'novel_name': novel_name,
        'character_name': character_name,
        'character_description': character_description,
        'image_quality': image_quality,
        'image_save_path': image_save_path,
        'status': 'Success' if response is not None else 'Failure'
    }

    if response is not None:
        image_id = response.data[0].url
        save_path = save_image(image_id, image_save_path)
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
    parser.add_argument('--image_save_path', type=str, help='The path to save the image.', default='./images/default.png')
    parser.add_argument('--log_file_path', type=str, help='The path to the log file.', default='./log.csv')
    parser.add_argument('--character_description', type=str, help='The description of the character.', default='A person with a cat tail and cat ears.')
    args = parser.parse_args()

    fetch_and_save_dalle_image_with_logging(args.novel_name, args.character_name, args.character_description,args.api_key_file_path, args.image_save_path, args.log_file_path)
