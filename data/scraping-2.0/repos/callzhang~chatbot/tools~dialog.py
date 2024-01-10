# Description: dialog management
import os, logging, json, re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from . import model, utils, google_sheet
import streamlit as st
from gspread import Worksheet, Spreadsheet, utils as gutils
from gspread_pandas import Spread
from streamlit.runtime.uploaded_file_manager import UploadedFile
from functools import lru_cache
from dateutil.parser import parse
from retry import retry

# 管理对话历史，存储在云端Google Drive里面
# 文件夹：·CHAT_FOLDER·
CHAT_FOLDER = 'chatbot'
# 里面每个spreadsheet对应每个用户，表格名称和用户名相同
# 每个表格里面有一个history sheet， header为[time, title, sheet]
HISTORY_SHEET_NAME = 'history'
HISTORY_HEADER = ['time', 'title', 'sheet']
# 其余sheet为dialog，名称对应history的sheet
DIALOG_HEADER = ['role', 'name', 'content', 'time', 'task', 'suggestions', 'actions', 'medias', 'status']
'''Objects 对应关系
| dataclass   |   model     | cloud       | comment  |
|     -       |   dialog    | spreadsheet | 所有对话  |
|[AppMessage] | conversation| sheet       | 对话      |
| AppMessage  |  message    |   row       | 消息      |
'''

client = google_sheet.init_client()

# init prompt
system_prompt = [
    {"role": "system", "content": f"你是星尘小助手，Your name is Stardust AI Bot. 你是由星尘数据的CEO Derek创造的，你的底层是基于Transformer的技术研发。你会解答各种AI专业问题，如果你不能回答，请让用户访问“stardust.ai”。今天是{datetime.now()}"},
    {"role": "system", "content": "星尘数据（Stardust）成立于2017年5月，公司在北京，是行业领先的数据标注和数据策略公司。星尘数据将专注AI数据技术，通过Autolabeling技术、数据策略专家服务和数据闭环系统服务，为全球人工智能企业特别是自动驾驶行业和大模型应用落地提供“燃料”，最终实现AI的平民化。"},
]

suggestion_prompt = {"role": "system", "content": f'请在你的回答的最后面给出3个启发性问题，让用户可以通过问题进一步理解该概念，并确保用户能继续追问。格式格式为：{utils.SUGGESTION_TOKEN}: ["启发性问题1", "启发性问题2", "启发性问题3"]。请注意：这个启发性问题列表放在最后，且用一行展示，不要换行。'}
search_prompt = {"role": "system", "content": f'如果用户的问题是常识性的问题，请直接回答，不用调用function检索。今天是{datetime.now()}'}

staff_prompt = lambda user: [{"role": "assistant", "content": f"你好，{user}，请问有什么可以帮助你？"}]
guest_prompt = lambda user: [{"role": "system", "content": f'用户是访客，名字为{user}，请用非常精简的方式回答问题。'},
                             {'role': 'assistant', 'content': f'欢迎您，{user}！'}]
TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


# dialog: all conversations
def init_dialog_history(username):
    # dialog_history: 所有对话标题的索引，[time, title, file]
    # conversation: 对话的具体内容列表，[{role, name, time, content, suggestion},...]
    
    # 初始化当前对话
    history = get_dialog_history(username)
    # 初始化对话列表
    st.session_state.dialog_history = history.col_values(2)[1:]
    if not st.session_state.dialog_history:
        # 没有历史记录或创建新对话，增加“新对话”至title
        dialog_title = new_dialog(username)
        st.session_state.selected_title = dialog_title
        st.rerun()
    elif 'new_title' in st.session_state:
        # 点击新增dialog按钮，通过“new_title”来传递逻辑
        st.session_state.selected_title = st.session_state.new_title
        del st.session_state.new_title
    elif 'chat_title_selection' in st.session_state:
        # select the title according to "chat_title_selection" UI selection
        st.session_state.selected_title = st.session_state.chat_title_selection
    # if current selected title in UI doesn't exist (due to deletion), select a new title
    if 'selected_title' not in st.session_state or st.session_state.selected_title not in st.session_state.dialog_history:
        logging.warning(
            f'Current selected title {st.session_state.get("selected_title")} not in history ({st.session_state.dialog_history}), select a new one')
        st.session_state.selected_title = st.session_state.dialog_history[0]
    # load conversation
    if "conversation" not in st.session_state:
        load_conversation(username, st.session_state.selected_title)
        
def load_conversation(username, title):
    # get对话记录
    messages = get_messages(username, title)
    st.session_state.conversation = messages
    return messages


## message
class Chat(model.Message):
    '''Chat object for streamlit UI'''
    pass
    
@utils.run_in_thread
def update_message(username, title, message:model.Message, create=False):
    from .controller import openai_image_types
    dialog = get_dialog(username, title)
    # create chat entry as a dict
    message_dict = message.model_dump() 
    message_dict = {k:v for k ,v in message_dict.items() if k in DIALOG_HEADER}
    assert message_dict, f'Empty chat: {message}'
    # convert medias to local file
    if message.medias:
        media_uri_list = [m._file_urls for m in message.medias]
        first_url = media_uri_list[0]
        filename, mime_type = utils.parse_file_info(first_url)
        if len(media_uri_list) == 1 and mime_type.split('/')[-1] in openai_image_types:
            message_dict['medias'] = f'=IMAGE("{first_url}")'
        else:
            message_dict['medias'] = media_uri_list
    message_value = convert_to_cell_values(message_dict)
    if create:
        res = dialog.append_row(message_value, value_input_option='USER_ENTERED')
    else:
        try:
            time_str = message.time.strftime('%Y-%m-%d %I:%M:%S')
            time_col = DIALOG_HEADER.index('time')+1
            row_index = dialog.find(time_str, in_column=time_col).row
        except Exception as e:
            # 没有找到匹配的时间，尝试找最近的时间
            times = [parse(t) for t in dialog.col_values(time_col)[1:]]
            t0 = min(times, key=lambda t: abs(message.time-t))
            if abs(message.time-t0) < timedelta(seconds=10):
                row_index = times.index(t0) + 2
                # logging.info(f'time matched: {message.time}<->{t0}')
            else:
                logging.error(f'Cannot find message: {time_str}<->{dialog.col_values(time_col)}')
                return
        row_values = dialog.row_values(row_index)
        if row_values[0] != message_value[0]:
            logging.error(f'Value mismatch: \n{row_values}\n{message_value}')
            return
        # we cannot update the whole row, google api will raise error, probably due to large content of `status`
        from itertools import zip_longest
        for i, (v0, v1) in enumerate(zip_longest(row_values, message_value)):
            if v0 != v1 and i != 3 and v1:
                dialog.update_cell(row=row_index, col=i+1, value=v1)


# get messages from a dialog
def get_messages(username, title):
    '''get messages used for UI'''
    dialog_sheet = get_dialog(username, title)
    records = dialog_sheet.get_records(value_render_option='FORMULA')
    # convert to Message object
    messages = []
    for c in records:
        try:
            msg = model.Message(**c)
            messages.append(msg)
        except Exception as e:
            logging.error(f'Error when loading chat: {c} \n Error: {e}')
    return messages

@utils.run_in_thread
@retry(3, delay=1, backoff=2)
def delete_message(username, title, row_num=None):
    dialog = get_dialog(username, title)
    row_num = row_num or dialog.row_count
    dialog.delete_rows(row_num)
    
    

# dialog
@utils.cached(timeout=7200)
def get_dialog(username:str, title:str) -> Worksheet:
    '''Find the dialog file from history
    :param username: the user to search for
    :param title: the title of the dialog
    :returns: the sheet object contains the whole dialog
    '''
    history = get_dialog_history(username)
    cell = history.find(title, in_column=2)
    dialog_title = history.cell(cell.row, 3).value
    all_sheets = [s.title for s in history.spreadsheet.worksheets()]
    if dialog_title in all_sheets:
        dialog = history.spreadsheet.worksheet(dialog_title)
    else:
        new_dialog(username, dialog_title)
        dialog = history.spreadsheet.worksheet(dialog_title)
    return dialog


@utils.cached(timeout=7200)
def get_dialog_history(username) -> Worksheet:
    # history_file = CHAT_LOG_ROOT/username/'history.csv'
    record_file = Spread(username, sheet='history', client=client, create_sheet=True, create_spread=True)
    # history = history.sheet_to_df(index=None, formula_columns=['medias'])
    if not record_file.sheet.get_values():
        sh1 = record_file.spread.worksheet('Sheet1')
        record_file.spread.del_worksheet(sh1)
        record_file.sheet.append_row(HISTORY_HEADER)
        record_file.spread.share('leizhang0121@gmail.com', perm_type='user', role='writer')
    return record_file.sheet


def new_dialog(username, dialog_title=None) -> str:
    now = datetime.now()
    if not dialog_title:
        dialog_title = now.strftime(TIME_FORMAT)
    history = get_dialog_history(username)
    all_historys = history.col_values(2)[1:]
    if dialog_title in history.col_values(2):
        all_sheets = [s.title for s in history.spreadsheet.worksheets()]
        if dialog_title in all_sheets:
            print(f'dialog title {dialog_title} exists!')
            return dialog_title
    else:
        row = [now.isoformat(), dialog_title, dialog_title]
        if all_historys:
            history.insert_row(row, index=2, value_input_option='USER_ENTERED')
        else:
            history.append_row(row, value_input_option='USER_ENTERED')
    # create sheet
    new_dialog = history.spreadsheet.add_worksheet(dialog_title, 1, 1)
    new_dialog.append_row(DIALOG_HEADER)
    # update system prompt
    conversation = system_prompt.copy()
    if st.session_state.guest:
        conversation += guest_prompt(username)
    else:
        conversation += staff_prompt(username)
    # update to sheet
    conversation_df = pd.DataFrame(conversation)
    conversation_df['time'] = datetime.now()
    conversation_df['task'] = None
    conversation_df['name'] = 'system'
    records = conversation_df.to_dict(orient='records')
    message_values = [convert_to_cell_values(r) for r in records]
    new_dialog.append_rows(message_values, value_input_option='USER_ENTERED')
    return dialog_title
    
@utils.run_in_thread(callback=lambda _: st.rerun())
def edit_dialog_title(username, old_title, new_title):
    history = get_dialog_history(username)
    cell = history.find(old_title, in_column=2)
    history.update_cell(row=cell.row, col=cell.col, value=new_title)
    st.session_state.new_title = new_title
    return new_title
    

@utils.run_in_thread(callback=lambda _: st.rerun())
def delete_dialog(username, title):
    history = get_dialog_history(username)
    cell = history.find(title, in_column=2)
    # sheet = history.cell(row=cell.row, col=3).value
    # dialog = history.spreadsheet.worksheet(sheet)
    # history.spreadsheet.del_worksheet(dialog)
    history.delete_rows(cell.row)
    
# utils
def convert_to_cell_values(record: dict):
    message_value = [str(record[k]) if k in record and record[k] else '' for k in DIALOG_HEADER]
    return message_value


## -------- assistant session ---------
# THREAD_FILE = 'threads.csv'
# def get_threads(username):
#     threads_history_file = CHAT_LOG_ROOT/username/THREAD_FILE
#     if os.path.exists(threads_history_file):
#         history = pd.read_csv(threads_history_file, index_col=0)
#     else:
#         history = pd.DataFrame(columns=['time', 'title', 'thread_id', 'assistant_id'])
#     return history


# def new_thread(username, thread_id, assistant_id, title=None):
#     history = get_threads(username)
#     new_dialog = pd.DataFrame([{
#         'time': datetime.now(),
#         'title': title if title else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'thread_id': thread_id,
#         'assistant_id': assistant_id,
#     }])
#     history = pd.concat([new_dialog, history], ignore_index=True)
#     os.makedirs(CHAT_LOG_ROOT/username, exist_ok=True)
#     history.to_csv(CHAT_LOG_ROOT/username/THREAD_FILE)
#     return title


# def delete_thread(username, thread_id):
#     history = get_dialog_history(username)
#     chat = history.query('thread_id==@thread_id')
#     history.drop(chat.index.values, inplace=True)
#     history.to_csv(CHAT_LOG_ROOT/username/THREAD_FILE)


## ----------- Utils -----------
# export
# 导出对话内容到 markdown
def conversation2markdown(messages:list[model.Message], title=""):
    if not messages:
        return ''
    conversation = [m.model_dump() for m in messages]
    history = pd.DataFrame(conversation).query('role not in ["system", "audio"]')
    # export markdown
    md_formated = f"""# 关于“{title}”的对话记录
*导出日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n
"""
    for i, c in history.iterrows():
        role, content, task, name, time = c['role'], c['content'], c.get('task'), c.get('name'), c.get('time')
        if role == "user":
            md_formated += '---\n'
            md_formated += f"""**[{time}]{name}({task}): {content}**\n\n"""
        elif role in ["assistant"]:
            md_formated += f"""星尘小助手({name}): {content}\n\n"""
        elif role == "DALL·E":
            md_formated += f"""星尘小助手({name}): {content}\n\n"""
        elif role == '':
            pass
        else:
            raise Exception(f'Unhandled chat: {c}')
    return md_formated.encode('utf-8').decode()

