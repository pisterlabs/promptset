import streamlit as st
import openai
import toml
import json
import numpy as np


with open('secrets.toml', 'r') as f:
    config = toml.load(f)

openai.api_type = "azure"
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = config['OPENAI_API_BASE']
openai.api_version = "2023-07-01-preview"

# Load JSON Data
with open('./data/json_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Streamlit Configuration
st.set_page_config(
    page_title="Home",
    page_icon="🚗",
)
st.header("歡迎來到汽車保險管理中心 。🚗")
st.subheader('筆錄案例 #37294810', '📞')

transcript = "客服人員：早上好，謝謝您致電汽車保險公司，我是John，今天我可以如何幫助您？\n客戶：是的，你好，我剛剛發現我的車側面有一個凹痕，我不知道怎麼回事。附近沒有目擊者，我真的很沮喪。\n客服人員：聽到這個消息我很抱歉，我理解這有多令人沮喪。您能提供姓名和保單號碼，讓我查看您的帳戶資訊嗎？\n客戶：是的，我是Mario Rossi，保單號碼是123456。\n客服人員：感謝您，Rossi先生，讓我查一下。我看到您今天早些時候已經打過電話了，那次通話有問題嗎？\n客戶：是的，我被擱置了超過一個小時，問題還沒有解決。我真的對此不滿意。\n客服人員：對此我深感抱歉，讓我向您保證，我們重視您的時間，今天會盡我們所能協助您。至於您車上的凹痕，我想通知您我們的保單確實涵蓋這種意外損壞。我可以幫您提出索賠，並將您介紹到我們信賴的修車行。您滿意這個結果嗎？\n客戶：是的，請這麼做。那真的很棒。\n客服人員：感謝您的合作。我正在處理您的索賠，並將向您發送一封帶有後續步驟的電子郵件。請告訴我是否還有其他問題或擔憂。\n客戶：謝謝您，我很感謝您的幫助。\n客服人員：不客氣。祝您有美好的一天！\n\n\n"
st.text(transcript)


# OpenAI Response Function
def openai_response(chat):
    system_message = "You are a helpful assistant with deep contract knowledge. Answer accurately or say you don't know. Respond in traditional Chinese."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(data) + chat}
    ]

    response = openai.ChatCompletion.create(
        engine="gpt-4-32k",
        messages=messages,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95
    )
    
    return response['choices'][0]['message']['content'].strip()

# UI Components
if st.button('建立支援工單'):
    ticket_number = np.random.randint(1, 1000000)
    st.write(f'您的工單已經創建，編號為 {ticket_number}。客戶和事件管理員將很快收到通知。')

if st.button('產生Email'):
    chat = f"生成一封回應上述筆錄的電子郵件，通知客戶已創建了工單，並且如果是投訴，則表示歉意。客戶的名字是 {data['客戶姓名']}，保單號碼是 {data['保單號碼']}。"
    st.write(openai_response(chat))

if st.button('改善客服品質'):
    chat = f"制定一個改善措施清單，以達到以下改進：{data['聯絡中心改進的地方']}。"
    st.write(openai_response(chat))