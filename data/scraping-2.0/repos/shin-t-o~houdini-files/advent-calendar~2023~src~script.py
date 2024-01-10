from PySide2 import QtCore
from PySide2 import QtWidgets
import openai
import os
import time
import json


class ChatGptWorker(QtCore.QThread):
    # シグナル定義
    completed = QtCore.Signal(list)

    def __init__(self, input_text):
        QtCore.QThread.__init__(self)
        self.input_text = input_text

    def run(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI()

        # スレッド初期化
        thread = client.beta.threads.create()
        # スレッド作成
        message = client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=self.input_text
        )
        # Assistantの実行
        run = client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=os.getenv("ASSISTANT_ID")
        )

        # Runのステータスを監視 ... 履歴管理も含め、本来ならサーバでの実行を推奨
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            time.sleep(1)  # 1秒ごとにステータスを確認

        # メッセージの取得
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # メッセージの中からassistantの返答を抽出
        for msg in messages.data:
            if msg.role == "assistant":
                self.completed.emit(json.loads(msg.content[0].text.value))


class ChatGptWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        # whole menu
        self.setGeometry(500, 300, 450, 200)
        self.setWindowTitle("SOP node search widget")
        hbox = QtWidgets.QHBoxLayout(self)

        # chat input + send button
        left_vbox = QtWidgets.QVBoxLayout()
        # - chat input
        self.input = QtWidgets.QPlainTextEdit()
        self.input.move(20, 20)
        # - send button
        self.button = QtWidgets.QPushButton("Search node via GPT", self)
        self.button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.button.move(20, 100)
        # combine
        left_vbox.addWidget(self.input)
        left_vbox.addWidget(self.button)

        # - list menu
        self.list_widget = QtWidgets.QListWidget()
        # (input + button) + list
        hbox.addLayout(left_vbox)
        hbox.addWidget(self.list_widget)
        hbox.setStretchFactor(left_vbox, 2)
        hbox.setStretchFactor(self.list_widget, 1)

        self.setLayout(hbox)
        self.connect(self.button, QtCore.SIGNAL("clicked()"), self.exec_chatgpt)
        self.list_widget.itemClicked.connect(self.list_item_clicked)

    def exec_chatgpt(self):
        input_text = self.input.toPlainText()
        if input_text:
            # 質問内容を元にワーカー起動
            self.worker = ChatGptWorker(input_text)
            self.worker.completed.connect(self.update_list_widget)
            self.worker.start()

    def update_list_widget(self, responses):
        self.list_widget.clear()
        for res in responses:
            item = QtWidgets.QListWidgetItem(res)
            self.list_widget.addItem(item)

    def list_item_clicked(self, item):
        selected_node = item.text()
        # 何かしらのロジックでgeoノードを取得
        geo = hou.node(GEOMETRY_PATH)
        geo.createNode(selected_node)


dialog = ChatGptWidget()
dialog.show()
