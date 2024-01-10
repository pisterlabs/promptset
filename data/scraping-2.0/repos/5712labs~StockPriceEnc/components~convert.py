##########################################################################
### 공통함수 ###############################################################
##########################################################################
# streamlit_app.py
import streamlit as st
import openai
import pandas as pd
import msoffcrypto
import io
import tiktoken

def check_password():
    #----------------------Hide Streamlit footer----------------------------
    hide_st_style = """
        <style>
        #MainMenu {visibility: show;}
        header {visibility: show;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    #--------------------------------------------------------------------

    """Returns `True` if the user had the correct password."""
    def password_entered():
        # st.write(st.session_state)
        if "password_correct" in st.session_state:
           if st.session_state["password_correct"]:
               return True
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # st.session_state["openai_model"] = st.sidebar.selectbox("Choose a model:", ("gpt-3.5-turbo", "gpt-4"))
        # st.session_state["openai_key"] = st.sidebar.selectbox("Choose a model:", ("personal", "company"))
        # if st.session_state["openai_key"] == 'company':
        #     openai.api_key = st.secrets["api_dw"]
        # else:
        #     openai.api_key = st.secrets["api_key"]
        openai_options = ["gpt-3.5-turbo personal", "gpt-3.5-turbo company", "gpt-4 company"]
        st.session_state["openai_option"] = st.sidebar.selectbox("Choose a model:", openai_options, index=0)
        if st.session_state["openai_option"] == 'gpt-3.5-turbo personal':
            # openai.api_key = st.secrets["api_key"]
            openai.api_key = st.secrets["api_dw_oh"]
            st.session_state["openai_key"] = 'personal'
            st.session_state["openai_model"] = 'gpt-3.5-turbo'
        elif st.session_state["openai_option"] == 'gpt-3.5-turbo company':
            openai.api_key = st.secrets["api_dw"]
            st.session_state["openai_key"] = 'company'
            st.session_state["openai_model"] = 'gpt-3.5-turbo'
        elif st.session_state["openai_option"] == 'gpt-4 company':
            openai.api_key = st.secrets["api_dw"]
            st.session_state["openai_key"] = 'company'
            st.session_state["openai_model"] = 'gpt-4'

        return True

def get_kor_amount_string_no_change(num_amount, ndigits_keep):
    """잔돈은 자르고 숫자를 자릿수 한글단위와 함께 리턴한다 """
    return get_kor_amount_string(num_amount, 
                                 -(len(str(num_amount)) - ndigits_keep))

def get_kor_amount_string(num_amount, ndigits_round=0, str_suffix='원'):
    """숫자를 자릿수 한글단위와 함께 리턴한다 """
    assert isinstance(num_amount, int) and isinstance(ndigits_round, int)
    assert num_amount >= 1, '최소 1원 이상 입력되어야 합니다'
    ## 일, 십, 백, 천, 만, 십, 백, 천, 억, ... 단위 리스트를 만든다.
    maj_units = ['만', '억', '조', '경', '해', '자', '양', '구', '간', '정', '재', '극'] # 10000 단위
    units     = [' '] # 시작은 일의자리로 공백으로하고 이후 십, 백, 천, 만...
    for mm in maj_units:
        units.extend(['십', '백', '천']) # 중간 십,백,천 단위
        units.append(mm)
    
    list_amount = list(str(round(num_amount, ndigits_round))) # 라운딩한 숫자를 리스트로 바꾼다
    list_amount.reverse() # 일, 십 순서로 읽기 위해 순서를 뒤집는다
    
    str_result = '' # 결과
    num_len_list_amount = len(list_amount)
    
    for i in range(num_len_list_amount):
        str_num = list_amount[i]
        # 만, 억, 조 단위에 천, 백, 십, 일이 모두 0000 일때는 생략
        if num_len_list_amount >= 9 and i >= 4 and i % 4 == 0 and ''.join(list_amount[i:i+4]) == '0000':
            continue
        if str_num == '0': # 0일 때
            if i % 4 == 0: # 4번째자리일 때(만, 억, 조...)
                str_result = units[i] + str_result # 단위만 붙인다
        elif str_num == '1': # 1일 때
            if i % 4 == 0: # 4번째자리일 때(만, 억, 조...)
                str_result = str_num + units[i] + str_result # 숫자와 단위를 붙인다
            else: # 나머지자리일 때
                str_result = units[i] + str_result # 단위만 붙인다
        else: # 2~9일 때
            str_result = str_num + units[i] + str_result # 숫자와 단위를 붙인다
    str_result = str_result.strip() # 문자열 앞뒤 공백을 제거한다 
    if len(str_result) == 0:
        return None
    if not str_result[0].isnumeric(): # 앞이 숫자가 아닌 문자인 경우
        str_result = '1' + str_result # 1을 붙인다
    return str_result + str_suffix # 접미사를 붙인다

def calculate_rsi(data, window_length=14):
    data = data.copy()
    delta = data['Close'].diff()
    delta = delta[1:] 

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    avg_gain = up.rolling(window_length).mean()
    avg_loss = abs(down.rolling(window_length).mean())

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    return data

def get_df_from_password_excel(excelpath, password):
    df = pd.DataFrame()
    temp = io.BytesIO()
    with open(excelpath, 'rb') as f:
        excel = msoffcrypto.OfficeFile(f)
        excel.load_key(password)
        excel.decrypt(temp)
        df = pd.read_excel(temp)
        del temp
    return df

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

