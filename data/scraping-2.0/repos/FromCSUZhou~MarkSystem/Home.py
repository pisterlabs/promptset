import streamlit as st
import openai

import json
import requests
import datetime

api_keys = st.secrets["API_KEYS"]
openai.api_key = api_keys
MONGO_URL_findOne = st.secrets["MONGO_URL_findOne"]
MONGO_URL_updateOne = st.secrets["MONGO_URL_updateOne"]
MONGO_KEY = st.secrets["MONGO_KEY"]


def find_one(user_key):
    payload = json.dumps({
        "collection": "AccountInfo",
        "database": "SZRI",
        "dataSource": "ChatgptUsing",
        "filter": {"email_address": user_key},
    })
    headers = {
                  'Content-Type': 'application/json',
                  'Access-Control-Request-Headers': '*',
                  'api-key': MONGO_KEY,
    }
    response = requests.request("POST", MONGO_URL_findOne, headers=headers, data=payload)
    print(response.text)
    return json.loads(response.text)


def email_check(user_key, check_date=True):
    email_data = find_one(user_key)
    if email_data["document"] and check_date:
        expiry_date_str = email_data["document"]["expiry_date"]
        expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()

        # 判断 expiry_date 是否在今天之前
        if expiry_date >= datetime.date.today():
            return True
        else:
            return False
    elif email_data["document"] and not check_date:
        return True
    else:
        return False

st.set_page_config(page_title="SZRI打标系统", page_icon=":guardsman:", layout="wide")

user_key = st.text_input("请在这儿输入您的邮箱后按回车键～")

if email_check(user_key):
    user_info = find_one(user_key)
    expire_date = user_info["document"]["expiry_date"]
    # used_tokens = user_info["document"]["used_tokens"]
    st.success("""验证成功，欢迎使用SZRI打标系统 👋\n
    您的有效期限至：{}""".format(expire_date))
    if "shared" not in st.session_state:
        st.session_state["shared"] = True
    if "user_key" not in st.session_state:
        st.session_state["user_key"] = user_key
else:
    st.error("请输入您的邮箱或检查您的邮箱是否正确～")


st.info("""
实验账号：checker@hku.hk; labeler@hku.hk \n
TodoList:\n
- [ ] 1. 分别测试打标系统不同页面并优化\n
- [ ] 2. 插入打标系统使用说明\n
- [ ] 3. 关于RLHF页面，需要等模型微调好之后再接入模型\n
权限说明：checker > labeler \n
checker比labeler多一个页面展示（数据审核页面）\n
\n
SZRI
""")
