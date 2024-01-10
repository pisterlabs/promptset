# -*- coding:utf-8 -*-
# title           :chat_window_worker.py
# description     :聊天窗口线程访问worker
# author          :Python超人
# date            :2023-6-3
# link            :https://gitcode.net/pythoncr/
# python_version  :3.8
# ==============================================================================

from PyQt5.QtCore import pyqtSignal, QObject

from common.chat_utils import ERR_MSG_MAP, message_to_html
from common.openai_chatbot import OpenAiChatbot
from db.db_ops import HistoryOp
from windows.chat_window import ChatWindow


class ChatWindowWorker(QObject):
    """
    聊天窗口线程访问worker
    """
    appendTitleSignal = pyqtSignal(int, bool, str, str, str)
    updateHtmlSignal = pyqtSignal(str)
    finishSignal = pyqtSignal()

    def __init__(self, ui: ChatWindow):
        super(ChatWindowWorker, self).__init__()
        self.ui = ui
        self.chat_stop = False

    def do(self):
        try:
            self._do()
        except Exception as e:
            self.ui.session_events.on_recieved_message(0, str(e))

    def _do(self):
        his_id = HistoryOp.insert(role="assistant",
                                  content='',
                                  content_type="text",
                                  session_id=self.ui.session_id,
                                  status=0)

        self.appendTitleSignal.emit(his_id, True, "icon_green.png", "OpenAi", "green")
        self.updateHtmlSignal.emit('<div class="loading"></div>')

        content = ""
        reply_error = 0

        for reply, status, is_error in OpenAiChatbot().chat_messages(self.ui.messages, self.ui.model_id):
            if reply_error != 1:
                reply_error = is_error
            part_content = reply["content"]
            content += part_content
            html = message_to_html(content)
            if self.chat_stop:
                break
            self.updateHtmlSignal.emit(html)

        self.chat_stop = False
        self.finishSignal.emit()

        if reply_error == 1:
            for e_key in ERR_MSG_MAP.keys():
                if e_key in content:
                    self.updateHtmlSignal.emit("<br>" + ERR_MSG_MAP[e_key])
                    break

        content_len = len(content)

        if reply_error == 0:
            reply = {"role": "assistant", "content": content}
            self.ui.chat_history.append((reply_error, reply, content_len))
            self.ui.session_events.on_chat_history_changed()
            HistoryOp.update_content(his_id, content)

        if reply_error == 1:  # 有问题
            pass

        self.ui.session_events.on_recieved_message(0, content)
