import copy
import importlib
import json
import re
import time

import openai
import streamlit as st


FUNC_PROMPT = """You are allowed to call external functions to finish tasks by output function calling starting with "```func" and ending with "```". Example: ```func\n{"name": "func_name", "parameters": {"param1": "abc", "p2": 123}}. You are not allowed to call non-listed functions. A response can contain at most a function call. The available functions are listed below."""



def delete_msg(session, idx):
    print('delete_msg', idx)
    session['history'].pop(idx)


def choose_session(chat_id):
    st.session_state['selected_session'] = chat_id


def send_msg(history: list, msg):

    history = copy.deepcopy(history)

    system_prompt = "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully."
    if len(st.session_state['enabled_functions']) > 0:
        system_prompt += FUNC_PROMPT

        function_prompt = []
        for func in st.session_state['enabled_functions']:
            function_prompt.append(st.session_state['enabled_functions'][func]['function_desc_for_model'])
        system_prompt += json.dumps(function_prompt)

    history.insert(0, {
        'role': 'system',
        'content': system_prompt,
    })

    history.append({
        'role': 'user',
        'content': msg,
    })

    print('send_msg', history)
    response = openai.ChatCompletion.create(
        model=st.session_state['model'],
        messages=history,
        stream=True
    )

    for chunk in response:
        if 'content' in chunk['choices'][0]['delta']:
            content = chunk['choices'][0]['delta']['content']
            yield content


def start_new_session():
    st.session_state['sessions'].insert(0, {
        'title': '新会话',
        'history': []
    })
    st.session_state['selected_session'] = 0


def not_implemented():
    st.toast('还未实现')


# 配置警告
def ui_settings_warning():
    if st.session_state['api_base'] == '':
        st.warning('API_Base为空，默认将使用OpenAI')
    else:
        openai.api_base = st.session_state['api_base']

    if st.session_state['api_key'] == '':
        st.error("API_KEY未设置，请去设置页面设置您的API_KEY")
    else:
        openai.api_key = st.session_state['api_key']


def ui_sidebar():

    with st.sidebar:
        st.button('开启新会话', type='primary', use_container_width=True, on_click=start_new_session)

        for i, session in enumerate(st.session_state['sessions']):
            if i == st.session_state['selected_session']:
                label = f':red[{session["title"]}]'
            else:
                label = session["title"]
            st.button(label, use_container_width=True, on_click=choose_session, args=(i,), key=f'session-{i}')


def ui_chat(session):
    # 聊天历史
    for idx, msg in enumerate(session['history']):
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            st.button('删除', key=f'delete-{idx}', on_click=delete_msg, args=(session, idx,))

    # 聊天输入框

    prompt = st.chat_input('说点什么吧...')
    if prompt:
        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
            result = ''
            with st.spinner('Thinking...'):
                msg = st.empty()
                for chunk in send_msg(session['history'], prompt):
                    result += chunk
                    msg.write(result)
                session['history'].append({
                    "role": "user",
                    "content": prompt,
                })
                session['history'].append({
                    "role": "assistant",
                    "content": result,
                })

        cnt = 0  # 用于计数，防止死循环，最多允许调用3次函数
        while True:
            cnt += 1

            func_call = re.findall(r'```func([\s\S]*)```', result)
            if func_call:
                if cnt > 3:
                    st.toast('调用函数次数过多，已停止调用. 如需继续，请输入"go on".')
                    break

                func_call = json.loads(func_call[0])
                func_name = func_call['name']
                with st.chat_message('function'):
                    if func_name in st.session_state['enabled_functions']:
                        func = st.session_state['enabled_functions'][func_name]
                        with st.spinner('函数运行中...'):
                            func_response = func['callable'](func_call['parameters'])
                            st.write('函数运行结果：' + func_response)
                            func_response = '函数执行结果：\n' + func_response
                    else:
                        func_response = 'This function is not available'
                with st.chat_message('assistant'):
                    with st.spinner('整合函数结果中：'):
                        msg = st.empty()
                        result = ''
                        for chunk in send_msg(session['history'], func_response):
                            result += chunk
                            msg.write(result)
                        session['history'].append({
                            "role": "function",
                            "content": func_response,
                        })
                        session['history'].append({
                            "role": "assistant",
                            "content": result,
                        })
            else:
                break



        st.experimental_rerun()

def main():
    if len(st.session_state['sessions']) == 0:
        start_new_session()
        st.experimental_rerun()

    session = st.session_state['sessions'][st.session_state['selected_session']]
    # 标题行
    st.title(f'{session["title"]}')
    cols = st.columns(2)
    cols[0].button('修改标题', use_container_width=True, on_click=not_implemented)
    cols[1].button('删除会话', use_container_width=True, on_click=not_implemented)

    ui_settings_warning()

    ui_sidebar()

    ui_chat(session)


if __name__ == '__main__':
    main()
