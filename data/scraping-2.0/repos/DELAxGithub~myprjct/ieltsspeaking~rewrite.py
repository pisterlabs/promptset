import os
import openai
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# OpenAIのAPIキーを設定します。
openai.api_key = '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'

def rewrite_text(original_text):
    rewrite_prompt = f"{original_text}\n\nRewrite this text:"
    message = {"role": "system", "content": "You are a helpful assistant that rewrites sentences."}, {"role": "user", "content": rewrite_prompt}
    response = openai.ChatCompletion.create(
      model="gpt-4",
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
sh = gc.open('answer1')
worksheet = sh.sheet1
generated_texts = worksheet.get_all_values()

# 各生成テキストに対してループを行います。
for i, row in enumerate(generated_texts):
    text = row[0]  # 生成テキストは各行の最初の要素とします。

    # 生成したテキストをリライトします。
    rewritten = rewrite_text(text)

    # リライトした結果を同じGoogle Sheetsに書き込みます（B列に）。
    worksheet.update_cell(i+1, 2, rewritten)
