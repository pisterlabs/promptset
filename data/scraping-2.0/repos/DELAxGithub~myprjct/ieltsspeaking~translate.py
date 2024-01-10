import os
import openai
import gspread
import time  # 追加
from oauth2client.service_account import ServiceAccountCredentials

# OpenAIのAPIキーを設定します。
openai.api_key = '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'

def translate_text(original_text):
    translate_prompt = f"{original_text}\n\nTranslate this text to Japanese:"
    message = {"role": "system", "content": "You are a helpful assistant that translates English to Japanese."}, {"role": "user", "content": translate_prompt}
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # ここを変更model="gpt-4",  
      messages=message
    )
    
    return response['choices'][0]['message']['content']

# Google SheetsとOpenAIの認証情報を設定します。
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/delax/Documents/Projects/ieltsspeaking/client_secret.json"

# Google Sheetsからデータを取得します。
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/delax/Documents/Projects/ieltsspeaking/client_secret.json', scope)
gc = gspread.authorize(creds)

# Google Sheetsから生成したテキストを読み込みます。
sh = gc.open('writing_expressions')
worksheet = sh.sheet1
original_texts = worksheet.get_all_values()

# 各原文に対してループを行います。
for i, row in enumerate(original_texts):
    text = row[0]  # 原文は各行の最初の要素とします。

    # 原文を翻訳します。
    translated = translate_text(text)

    # 翻訳した結果を同じGoogle Sheetsに書き込みます（B列に）。
    worksheet.update_cell(i+1, 2, translated)
    
    # 次のリクエストまで一定時間待機します。
    time.sleep(1)  # ここでは1秒待機しますが、必要に応じて調整してください。
