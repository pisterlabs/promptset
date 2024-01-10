import streamlit as st
import openai
import toml


with open('../secrets.toml', 'r') as f:
    config = toml.load(f)


openai.api_type = "azure"
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = config['OPENAI_API_BASE']
openai.api_version = "2023-07-01-preview"

st.set_page_config(
    page_title="Home",
    page_icon="📝",
)

st.header("歡迎使用合約分析器 📝")
st.subheader('合約 #371')

contract = """

此服務合約(「協議」)於[日期]由Company A(「公司」)和Company B(「服務供應者」)訂立。

1. 提供服務: 服務提供者同意為公司提供以下服務(「服務」)：服務提供者同意為公司提供市場營銷領域的諮詢服務，包括但不限於市場研究、營銷策略的制定和營銷活動的實施。服務提供者應根據市場研究的結果和經雙方同意的營銷策略，向公司提供報告和建議。
2. 支付金額: 公司應支付服務提供者1.000.000(一百萬)美元作為服務費用。支付日期為2023年9月15日。
3. 行使期限: 本協議自2023年5月1日開始，持續至2023年12月31日，除非任何一方提前30天書面通知終止。
4. 獨立承包: 服務提供者是一個獨立承包商，本協議中的任何內容均不得解釋為在雙方之間建立僱主-員工關係、合夥或合資企業。
5. 保密性: 服務提供者同意對於在為公司提供服務過程中學到或獲得的所有信息保密。未經公司事先書面同意，服務提供者不得向任何第三方披露此類信息。
6. 工作成果的所有權: 服務提供者同意，與服務相關的任何和所有工作成果均為公司的獨有財產。
7. 陳述和保證: 服務提供者陳述並保證其具有執行服務所需的專業知識和經驗，並將以專業和工匠的方式執行服務。
8. 賠償條款: 服務提供者同意賠償並使公司、其高級管理人員、董事、員工和代理人免受因服務而引起或與之相關的任何和所有索賠、損害、負債、成本和費用的損害。
9. 管轄法律: 本協議應受意大利法律的管轄並根據其解釋，不考慮法律衝突原則。
\n10 完整協議: 本協議構成雙方之間的完整協議，並取代雙方之間的所有先前或同時的談判、協議、陳述和理解，無論是書面還是口頭。

兹證明，雙方已於書面上述日期簽署本協議。

[公司簽名區塊]

[服務提供者簽名區塊]

"""

st.write(contract)

# Define a function to communicate with OpenAI using ChatCompletion
def openai_response(user_prompt):
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that has a deep understanding of contract. \
                                      You know all the detail about contract clauses.  \
                                      You will help people about their contract problems. \
                                      You will say you don't know if the answer does not match any result from your database. Be concise with your response. \
                                      Refrain from responding in simplified Chinese, you will respond in traditional Chinese at all time."
        },
        {"role": "user", "content": user_prompt}
    ]

    response = openai.ChatCompletion.create(
        engine="gpt-4-32k",
        messages=messages,
        temperature=0.7,
        max_tokens=8192,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    return response.choices[0].message['content'].strip()

st.subheader('關鍵條款提取 🔍')
# UI components
col1, col2 = st.columns(2)

with col1:
    request = st.selectbox(
        '選擇您要詢問的關鍵條款',
        ["終止條款是什麼？", "保密條款是什麼？", "支付金額是多少？", "到期日是什麼甚麼時候？", "賠償條款是什麼？"]
    )

with col2:
    if request:
        st.write(openai_response(contract + request))
        
        
# Language Analysis Section
st.subheader('其他問題 💬')
col3, col4 = st.columns(2)
with col3:
    user_input = st.text_input("You:", "")
with col4:
    if user_input:
        st.write(openai_response(contract + user_input))

# Potential Issues Section
st.subheader('潛在問題 🚩')
col5, col6 = st.columns(2)
with col5:
    request = st.selectbox(
        '選擇您要詢問的關鍵條款',
        ["合約中有模糊之處嗎？", "合約中有相互衝突的條款嗎？"]
    )
with col6:
    if request:
        st.write(openai_response(contract + request))

# Contract Template Section
st.subheader('合約模板產生器 🖋️')
col7, col8 = st.columns(2)
with col7:
    service_provider = st.text_input("服務供應商:", "")
    client = st.text_input("客戶:", "")
    services_description = st.text_input("服務描述:", "")
    start_date = st.text_input("開始日期:", "")
    duration = st.text_input("服務持續期間:", "")
with col8:
    if st.button('生成模板'):
        prompt_text = f"按照以下元素生成服務交付協議：服務提供者：{service_provider}，客戶：{client}，服務描述：{services_description}，開始日期：{start_date}，服務持續期間：{duration}。"
        st.write(openai_response(prompt_text))