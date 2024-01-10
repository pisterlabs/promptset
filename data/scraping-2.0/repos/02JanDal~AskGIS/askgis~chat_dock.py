from typing import Any, Optional

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QLabel, QLineEdit, QPlainTextEdit, QPushButton, QWidget
from qgis.core import QgsApplication, QgsProject
from qgis.gui import QgsDockWidget

from askgis.ask_dialog import get_api_key
from askgis.chat_task import ChatTask
from askgis.lib.context import compute_context
from askgis.qgis_plugin_tools.tools.resources import load_ui


def ui_file_dock(*ui_file_name_parts: str):  # noqa ANN201
    """DRY helper for building classes from a .ui file"""

    class UiFileDockClass(QgsDockWidget, load_ui(*ui_file_name_parts)):  # type: ignore
        def __init__(
            self,
            parent: Optional[QWidget],
        ) -> None:
            super().__init__(parent)
            self.setupUi(self)  # provided by load_ui FORM_CLASS

    return UiFileDockClass


DockUi = ui_file_dock("chat-dock-widget.ui")  # type: Any


class SignalingChatMessageHistoryObject(QObject):
    userMessage = pyqtSignal(str)
    aiMessage = pyqtSignal(str)
    clear = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)


class SignalingChatMessageHistory(ChatMessageHistory):
    obj: SignalingChatMessageHistoryObject

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, parent, **kwargs):
        ChatMessageHistory.__init__(
            self, *args, obj=SignalingChatMessageHistoryObject(parent), **kwargs
        )

    def add_user_message(self, message: str) -> None:
        super().add_user_message(message)
        self.obj.userMessage.emit(message)

    def add_ai_message(self, message: str) -> None:
        super().add_user_message(message)
        self.obj.aiMessage.emit(message)

    def clear(self) -> None:
        super().clear()
        self.obj.clear.emit()


class ChatDock(DockUi):
    clearBtn: QPushButton
    messageEdit: QLineEdit
    chatEdit: QPlainTextEdit
    sendBtn: QPushButton

    def __init__(self, parent: Optional[QWidget]):
        super().__init__(parent)

        self.sendBtn.clicked.connect(self.send)
        self.messageEdit.textChanged.connect(self.message_changed)
        self.message_changed()

        history = SignalingChatMessageHistory(parent=self)
        self._memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=history
        )
        history.obj.aiMessage.connect(self.handle_ai_message)
        history.obj.userMessage.connect(self.handle_user_message)
        history.obj.clear.connect(self.handle_clear)

        self.clearBtn.clicked.connect(history.obj.clear)

        self._task: Optional[ChatTask] = None

    def handle_ai_message(self, message: str):
        self.chatEdit.setPlainText(self.chatEdit.toPlainText() + f"AI: {message}\n")

    def handle_user_message(self, message: str):
        self.chatEdit.setPlainText(self.chatEdit.toPlainText() + f"You: {message}\n")

    def handle_clear(self):
        self.chatEdit.clear()

    def send(self):
        self.messageEdit.setDisabled(True)
        self.sendBtn.setDisabled(True)

        key = get_api_key(self)
        if key is None:
            self.messageEdit.setEnabled(True)
            self.message_changed()
            return

        context = compute_context(QgsProject.instance())

        self._task = ChatTask(
            self.tr("OpenAI"),
            self.messageEdit.text().strip(),
            context=context,
            api_key=key,
            memory=self._memory,
        )
        self._task.taskCompleted.connect(self.task_completed)
        QgsApplication.taskManager().addTask(self._task)

    def task_completed(self):
        self.messageEdit.clear()
        self.messageEdit.setEnabled(True)

    def message_changed(self):
        self.sendBtn.setEnabled(len(self.messageEdit.text().strip()) > 0)
