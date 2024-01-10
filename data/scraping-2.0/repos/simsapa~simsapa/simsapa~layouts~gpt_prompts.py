import re, json
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6.QtCore import QAbstractTableModel, QItemSelection, QItemSelectionModel, QModelIndex, QObject, QRunnable, QSize, QThreadPool, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QFont, QMovie, QStandardItem, QStandardItemModel
from functools import partial
from typing import Any, List, Optional, TypedDict
from datetime import datetime

import tiktoken

from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QMenuBar, QMessageBox,
                             QPushButton, QSpacerItem, QSpinBox, QSplitter, QTabWidget, QTableView, QTextEdit,
                             QTreeView, QVBoxLayout, QWidget)

from simsapa import IS_MAC, IS_SWAY, SEARCH_TIMER_SPEED, logger
from simsapa.app.export_helpers import sutta_content_plain
from simsapa.app.db import appdata_models as Am
from simsapa.app.db import userdata_models as Um

from simsapa.app.types import USutta
from simsapa.app.app_data import AppData

from simsapa.layouts.gui_types import (AppWindowInterface, ChatMessage, ChatResponse, ChatRole, OpenAIModel,
                                       OpenAIModelLatest, OpenAIModelToEnum, OpenAISettings, OpenPromptParams,
                                       QExpanding, QMinimum, default_openai_settings, model_max_tokens)

class ShowPromptDialog(QDialog):
    def __init__(self, text: str):
        super().__init__()

        self.setWindowTitle("Parsed Prompt Content")

        if IS_SWAY:
            self.setFixedSize(800, 600)
        else:
            self.resize(800, 600)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.prompt_text_input = QTextEdit()
        self.prompt_text_input.setPlainText(text)
        self.prompt_text_input.setFocus()
        self._layout.addWidget(self.prompt_text_input)

        self.buttons_box = QHBoxLayout()
        self._layout.addLayout(self.buttons_box)

        self.close_btn = QPushButton('Close')
        self.close_btn.clicked.connect(partial(self._handle_close))

        self.buttons_box.addWidget(self.close_btn)
        self.buttons_box.addItem(QSpacerItem(0, 0, QExpanding, QMinimum))

    def _handle_close(self):
        self.close()

# Keys with underscore prefix will not be shown in table columns.
HistoryModelColToIdx = {
    "Name": 0,
    "Prompt": 1,
    "Submitted": 2,
    "_db_id": 3,
}

class HistoryModel(QAbstractTableModel):
    def __init__(self, data = []):
        super().__init__()
        self._data = data
        self._columns = list(filter(lambda x: not x.startswith("_"), HistoryModelColToIdx.keys()))

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if len(self._data) == 0:
                return list(map(lambda _: "", self._columns))
            else:
                return self._data[index.row()][index.column()]
        elif role == Qt.ItemDataRole.UserRole:
            return self._data

    def rowCount(self, _):
        return len(self._data)

    def columnCount(self, _):
        if len(self._data) == 0:
            return 0
        else:
            return len(self._columns)

    def headerData(self, section, orientation, role):
       if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._columns[section]

            if orientation == Qt.Orientation.Vertical:
                return str(section+1)

class PromptData(TypedDict):
    db_id: int

class PromptItem(QStandardItem):
    name_path: str
    show_in_context: bool
    data: PromptData

    def __init__(self, name_path: str, show_in_context: bool, db_id: int):
        super().__init__()

        self.setEditable(False)

        # Remove trailing / for display.
        name_path = re.sub(r'/+$', '', name_path)

        self.name_path = re.sub(r' */ *', '/', name_path)

        self.show_in_context = show_in_context

        # Not storing db_schema, assuming all prompts are in userdata.
        self.data = PromptData(db_id = db_id)

        self.setEditable(False)
        self.setText(self.name_path)

class GptPromptsWindow(AppWindowInterface):
    _input_timer = QTimer()

    def __init__(self, app_data: AppData, prompt_params: Optional[OpenPromptParams] = None, parent = None) -> None:
        super().__init__(parent)
        logger.info("GptPromptsWindow()")

        self._app_data: AppData = app_data

        self.tokenizer_worker: Optional[TokenizerWorker] = None
        self.completion_worker: Optional[CompletionWorker] = None

        self.thread_pool = QThreadPool()

        self.sidebar_visible = True

        self._setup_ui()
        self._init_values()
        self._connect_signals()

        self._update_vert_splitter_widths()
        self._update_horiz_splitter_widths()

        if prompt_params is not None:
            self._show_prompt_by_params(prompt_params)

    def _setup_ui(self):
        self.setWindowTitle("GPT Prompts - Simsapa")
        self.resize(1068, 625)

        self._central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self._central_widget)

        self._layout = QVBoxLayout()
        self._central_widget.setLayout(self._layout)

        # horizontal splitter

        self.vert_splitter = QSplitter(self._central_widget)
        self.vert_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self._layout.addWidget(self.vert_splitter)

        self.left_box_widget = QWidget(self.vert_splitter)
        self.left_box_layout = QVBoxLayout(self.left_box_widget)
        self.left_box_layout.setContentsMargins(0, 0, 0, 0)

        self.right_box_widget = QWidget(self.vert_splitter)
        self.right_box_layout = QVBoxLayout(self.right_box_widget)
        self.right_box_layout.setContentsMargins(0, 0, 0, 0)

        # vertical splitter

        self.horiz_splitter = QSplitter(self._central_widget)
        self.horiz_splitter.setOrientation(QtCore.Qt.Orientation.Vertical)

        self.left_box_layout.addWidget(self.horiz_splitter)

        self.prompt_input_widget = QWidget(self.horiz_splitter)
        self.prompt_input_layout = QVBoxLayout(self.prompt_input_widget)
        self.prompt_input_layout.setContentsMargins(0, 0, 0, 0)

        self.completion_text_widget = QWidget(self.horiz_splitter)
        self.completion_text_layout = QVBoxLayout(self.completion_text_widget)
        self.completion_text_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.right_box_layout.addWidget(self.tabs)

        # bottom buttons

        self._bottom_buttons_box = QHBoxLayout()
        self._layout.addLayout(self._bottom_buttons_box)

        self.prompt_submit = QPushButton("Submit")
        self.prompt_submit.setMinimumSize(QSize(80, 40))
        self._bottom_buttons_box.addWidget(self.prompt_submit)

        self.openai_append_mode = QCheckBox("Append", self)
        self.openai_append_mode.setToolTip("Append the completion to the prompt")
        self._bottom_buttons_box.addWidget(self.openai_append_mode)

        label = QLabel("Model:")
        self._bottom_buttons_box.addWidget(label)

        self.openai_model_select = QComboBox()
        items = [i.value for i in OpenAIModel]
        self.openai_model_select.addItems(items)
        self.openai_model_select.setCurrentText(self._app_data.app_settings['openai']['model'])

        self._bottom_buttons_box.addWidget(self.openai_model_select)

        self.openai_temperature_input = QDoubleSpinBox()
        self.openai_temperature_input.setToolTip("Temperature")
        self.openai_temperature_input.setMinimum(0.0)
        self.openai_temperature_input.setMaximum(2.0)
        self.openai_temperature_input.setSingleStep(0.1)

        label = QLabel("T:")
        label.setToolTip("Temperature (random variation)")
        self._bottom_buttons_box.addWidget(label)
        self._bottom_buttons_box.addWidget(self.openai_temperature_input)

        self.openai_max_tokens_input = QSpinBox()
        self.openai_max_tokens_input.setToolTip("Max tokens to generate")
        self.openai_max_tokens_input.setMinimum(16)

        model_max = model_max_tokens(self._app_data.app_settings['openai']['model'])
        self.openai_max_tokens_input.setMaximum(model_max)

        label = QLabel("M:")
        label.setToolTip("Max tokens to generate")
        self._bottom_buttons_box.addWidget(label)
        self._bottom_buttons_box.addWidget(self.openai_max_tokens_input)

        self.openai_auto_max = QCheckBox("Auto max", self)
        self.openai_auto_max.setToolTip("Maximize generated token number")
        self._bottom_buttons_box.addWidget(self.openai_auto_max)

        self.token_count_msg = QLabel()
        self._bottom_buttons_box.addWidget(self.token_count_msg)

        self.token_warning_msg = QLabel(f"Warning: max total tokens for {self._app_data.app_settings['openai']['model']} is {model_max}")
        self._bottom_buttons_box.addWidget(self.token_warning_msg)
        self.token_warning_msg.setVisible(False)

        self.completion_loading_bar = QLabel()
        self._bottom_buttons_box.addWidget(self.completion_loading_bar)

        self._loading_bar_anim = QMovie(':loading-bar')
        self._loading_bar_empty_anim = QMovie(':loading-bar-empty')

        self.completion_warning_msg = QLabel()
        self._bottom_buttons_box.addWidget(self.completion_warning_msg)

        self._bottom_buttons_box.addItem(QSpacerItem(20, 0, QExpanding, QMinimum))

        self.toggle_sidebar_btn = QPushButton()

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/angles-right"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.toggle_sidebar_btn.setIcon(icon)
        self.toggle_sidebar_btn.setMinimumSize(QSize(40, 40))
        self.toggle_sidebar_btn.setToolTip("Toggle Sidebar")

        self._bottom_buttons_box.addWidget(self.toggle_sidebar_btn)

        self._setup_menubar()
        self._setup_editor()
        self._setup_prompts_tab()
        self._setup_history_tab()
        self._setup_settings_tab()

        self.reload_history_table()

    def _setup_menubar(self):
        self.menubar = QMenuBar()
        self.setMenuBar(self.menubar)

        self.menu_File = QMenu(self.menubar)
        self.menu_File.setTitle("&File")

        self.menubar.addAction(self.menu_File.menuAction())

        self.action_Close_Window = QAction("&Close Window")
        self.menu_File.addAction(self.action_Close_Window)

        self.action_Import = QAction("&Import from CSV...")
        self.menu_File.addAction(self.action_Import)

        self.action_Export = QAction("&Export as CSV...")
        self.menu_File.addAction(self.action_Export)

    def _setup_editor(self):
        if IS_MAC:
            font = QFont("Helvetica", pointSize = 13)
        else:
            font = QFont("DejaVu Sans", pointSize = 13)

        self.prompt_input_top_buttons_layout = QHBoxLayout()
        self.prompt_input_layout.addLayout(self.prompt_input_top_buttons_layout)

        self.prompt_clear_all_btn = QPushButton("Clear All")
        self.prompt_input_top_buttons_layout.addWidget(self.prompt_clear_all_btn)

        self.prompt_show_parsed_btn = QPushButton("Show Parsed")
        self.prompt_input_top_buttons_layout.addWidget(self.prompt_show_parsed_btn)

        self.prompt_input_top_buttons_layout.addItem(QSpacerItem(0, 0, QExpanding, QMinimum))

        self.prompt_input_top_buttons_layout.addWidget(QLabel("Copy:"))

        self.prompt_copy_btn = QPushButton("User Prompt")
        self.prompt_input_top_buttons_layout.addWidget(self.prompt_copy_btn)

        self.prompt_copy_completion_btn = QPushButton("Completion")
        self.prompt_input_top_buttons_layout.addWidget(self.prompt_copy_completion_btn)

        self.prompt_copy_all_btn = QPushButton("All")
        self.prompt_input_top_buttons_layout.addWidget(self.prompt_copy_all_btn)

        self.prompt_name_input = QLineEdit()
        self.prompt_name_input.setPlaceholderText("Prompt name, e.g. summarize text")
        self.prompt_input_layout.addWidget(self.prompt_name_input)

        self.prompt_input_layout.addWidget(QLabel("System:"))

        self.system_prompt_input = QTextEdit("You are a helpful assistant in understanding the teachings of the Buddha.")
        self.system_prompt_input.setFont(font)
        self.system_prompt_input.setMaximumHeight(100)
        self.prompt_input_layout.addWidget(self.system_prompt_input)

        self.prompt_input_layout.addWidget(QLabel("User:"))

        self.user_prompt_input = QTextEdit()
        self.user_prompt_input.setFont(font)
        self.prompt_input_layout.addWidget(self.user_prompt_input)

        self.user_prompt_input.setFocus()

        self.completion_text_layout.addWidget(QLabel("Completion:"))

        self.completion_text = QTextEdit()
        self.completion_text.setFont(font)
        self.completion_text_layout.addWidget(self.completion_text)

    def _setup_prompts_tab(self):
        self.prompts_tab_widget = QWidget()
        self.prompts_tab_layout = QVBoxLayout()
        self.prompts_tab_widget.setLayout(self.prompts_tab_layout)

        self.tabs.addTab(self.prompts_tab_widget, "Prompts")

        self.prompt_buttons_layout = QHBoxLayout()
        self.prompts_tab_layout.addLayout(self.prompt_buttons_layout)

        self.prompt_save_btn = QPushButton("Save Current")
        self.prompt_buttons_layout.addWidget(self.prompt_save_btn)

        self.prompt_toggle_menu_btn = QPushButton("Toggle Menu")
        self.prompt_buttons_layout.addWidget(self.prompt_toggle_menu_btn)

        self.prompt_buttons_layout.addItem(QSpacerItem(0, 0, QExpanding, QMinimum))

        self.prompt_delete_btn = QPushButton("Delete")
        self.prompt_buttons_layout.addWidget(self.prompt_delete_btn)

        self.prompt_delete_all_btn = QPushButton("Delete All")
        self.prompt_buttons_layout.addWidget(self.prompt_delete_all_btn)

        self._setup_prompts_tree_view()

    def _setup_prompts_tree_view(self):
        self.prompts_tree_view = QTreeView()
        self.prompts_tab_layout.addWidget(self.prompts_tree_view)

        self._init_prompts_tree_model()

    def _init_prompts_tree_model(self):
        self.prompts_tree_model = QStandardItemModel(0, 2, self)

        self.prompts_tree_view.setHeaderHidden(False)
        self.prompts_tree_view.setRootIsDecorated(True)

        self.prompts_tree_view.setModel(self.prompts_tree_model)

        self._create_prompts_tree_items(self.prompts_tree_model)

        # NOTE: Don't select the first item when opening the window. It is
        # confusing when the prompt is loaded from a sutta window.

        # idx = self.prompts_tree_model.index(0, 0)
        # self.prompts_tree_view.selectionModel() \
        #                       .select(idx,
        #                               QItemSelectionModel.SelectionFlag.ClearAndSelect | \
        #                               QItemSelectionModel.SelectionFlag.Rows)

        # self._handle_prompts_tree_clicked(idx)

        self.prompts_tree_view.expandAll()

    def reload_prompts_tree(self):
        self.prompts_tree_model.clear()
        self._create_prompts_tree_items(self.prompts_tree_model)
        self.prompts_tree_model.layoutChanged.emit()
        self.prompts_tree_view.expandAll()

    def _create_prompts_tree_items(self, model):
        item = QStandardItem()
        item.setText("Prompt")
        model.setHorizontalHeaderItem(0, item)

        item = QStandardItem()
        item.setText("Show in Context Menu")
        item.setToolTip("Show in Right-Click Context Menu")
        model.setHorizontalHeaderItem(1, item)

        root_node = model.invisibleRootItem()

        res = self._app_data.db_session \
                            .query(Um.GptPrompt) \
                            .order_by(Um.GptPrompt.name_path.asc()) \
                            .all()

        for r in res:
            do_show = (r.show_in_context is not None and r.show_in_context)
            s = "✓" if do_show else ""
            show = QStandardItem(s)

            prompt = PromptItem(r.name_path, do_show, r.id)

            root_node.appendRow([prompt, show])

        self.prompts_tree_view.resizeColumnToContents(0)
        self.prompts_tree_view.resizeColumnToContents(1)

    def _show_prompt_by_id(self, db_id: int):
        prompt = self._app_data.db_session \
                               .query(Um.GptPrompt) \
                               .filter(Um.GptPrompt.id == db_id) \
                               .first()
        if prompt is None or prompt.messages_json is None:
            return

        self.prompt_name_input.setText(prompt.name_path)

        messages: List[ChatMessage] = json.loads(prompt.messages_json)
        self._set_prompt_inputs(messages)

    def _set_prompt_inputs(self,
                           messages: List[ChatMessage],
                           parse_sutta_in_text = False,
                           sutta_uid: Optional[str] = None,
                           selection_text: Optional[str] = None):

        if len(messages) == 0:
            self.system_prompt_input.setPlainText("")
            self.user_prompt_input.setPlainText("")
            self.completion_text.setPlainText("")

            return

        if len(messages) > 0:
            self.system_prompt_input.setPlainText(messages[0]['content'])

        if len(messages) > 1:
            prompt = messages[1]['content']

            user_prompt = self._parse_prompt_variables(
                prompt = prompt,
                parse_sutta_in_text = parse_sutta_in_text,
                sutta_uid = sutta_uid,
                selection_text = selection_text)

            self.user_prompt_input.setPlainText(user_prompt)

        if len(messages) > 2:
            self.completion_text.setPlainText(messages[2]['content'])

    def _show_prompt_by_params(self, params: OpenPromptParams):
        prompt = self._app_data.db_session \
                               .query(Um.GptPrompt) \
                               .filter(Um.GptPrompt.id == params['prompt_db_id']) \
                               .first()
        if prompt is None or prompt.messages_json is None:
            return

        if params['with_name'] is None:
            self.prompt_name_input.setText(prompt.name_path)
        else:
            self.prompt_name_input.setText(params['with_name'])

        messages: List[ChatMessage] = json.loads(prompt.messages_json)
        self._set_prompt_inputs(messages, False, params['sutta_uid'], params['selection_text'])

    def _handle_prompts_tree_clicked(self, val: QModelIndex):
        item: PromptItem = self.prompts_tree_model.itemFromIndex(val) # type: ignore
        if item is not None:
            self._show_prompt_by_id(item.data['db_id'])

    def _setup_history_tab(self):
        self.history_tab_widget = QWidget()
        self.history_tab_layout = QVBoxLayout()
        self.history_tab_widget.setLayout(self.history_tab_layout)

        self.tabs.addTab(self.history_tab_widget, "History")

        self.history_buttons_layout = QHBoxLayout()
        self.history_tab_layout.addLayout(self.history_buttons_layout)

        self.history_load_btn = QPushButton("Load")
        self.history_buttons_layout.addWidget(self.history_load_btn)

        self.history_buttons_layout.addItem(QSpacerItem(0, 0, QExpanding, QMinimum))

        self.history_delete_btn = QPushButton("Delete")
        self.history_buttons_layout.addWidget(self.history_delete_btn)

        self.history_delete_all_btn = QPushButton("Delete All")
        self.history_buttons_layout.addWidget(self.history_delete_all_btn)

        self.history_table = QTableView()
        self.history_tab_layout.addWidget(self.history_table)

        self.history_table.setShowGrid(False)
        self.history_table.setWordWrap(False)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # MultiSelection allows multiple items to be selected with left-click,
        # and it becomes confusing what should be opened when the Open button or
        # double-click is used.
        self.history_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        horiz_header = self.history_table.horizontalHeader()
        if horiz_header:
            horiz_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            horiz_header.setStretchLastSection(True)

        self.history_table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        self.history_model = HistoryModel()
        self.history_table.setModel(self.history_model)

    def reload_history_table(self):
        data = self._data_items_for_history()

        self.history_model = HistoryModel(data)
        self.history_table.setModel(self.history_model)

        self.history_model.layoutChanged.emit()

    def _data_items_for_history(self) -> List[List[str]]:
        res = self._app_data.db_session \
                            .query(Um.GptHistory) \
                            .order_by(Um.GptHistory.created_at.desc()) \
                            .all()

        if len(res) == 0:
            return []

        def _model_data_item(x: Um.GptHistory) -> List[str]:
            # Return values ordered as in HistoryModelColToIdx
            if x.messages_json is None:
                text = ""
            else:
                m = json.loads(x.messages_json)
                if len(m) > 1:
                    text = m[1]['content'][0:20]
                else:
                    text = ""

            return ["" if x.name_path is None else str(x.name_path[0:20]),
                    text,
                    str(x.created_at),
                    str(x.id)]

        data = list(map(_model_data_item, res))

        return data

    def _setup_settings_tab(self):
        self.settings_tab_widget = QWidget()
        self.settings_tab_layout = QVBoxLayout()
        self.settings_tab_widget.setLayout(self.settings_tab_layout)

        self.tabs.addTab(self.settings_tab_widget, "Settings")

        self.settings_buttons_layout = QHBoxLayout()
        self.settings_tab_layout.addLayout(self.settings_buttons_layout)

        self.settings_buttons_layout.addItem(QSpacerItem(0, 0, QExpanding, QMinimum))

        self.settings_reset_btn = QPushButton("Reset")
        self.settings_buttons_layout.addWidget(self.settings_reset_btn)

        self.settings_tab_layout.addWidget(QLabel("OpenAI API key:"))
        self.openai_api_key_input = QLineEdit()
        self.openai_api_key_input.setPlaceholderText("sk-...")
        self.settings_tab_layout.addWidget(self.openai_api_key_input)

        self.openai_sign_up_info = QLabel("<p>Sign for an <a href='https://beta.openai.com/signup'>OpenAI account</a> and create your API key.</p>")
        self.openai_sign_up_info.setWordWrap(True)

        self.settings_tab_layout.addWidget(self.openai_sign_up_info)

        label = QLabel("<p>Number of completions to generate:</p>")
        label.setWordWrap(True)
        self.settings_tab_layout.addWidget(label)

        self.openai_n_completions_input = QSpinBox()
        self.openai_n_completions_input.setMinimum(0)
        self.openai_n_completions_input.setMaximum(10)
        self.openai_n_completions_input.setDisabled(True)

        self.settings_tab_layout.addWidget(self.openai_n_completions_input)

        label = QLabel("<p>Join short lines in the prompt under x chars to reduce token count:</p>")
        label.setWordWrap(True)
        self.settings_tab_layout.addWidget(label)
        self.openai_join_lines_under_input = QSpinBox()
        self.openai_join_lines_under_input.setMinimum(0)
        self.openai_join_lines_under_input.setMaximum(999)

        self.settings_tab_layout.addWidget(self.openai_join_lines_under_input)

        self.settings_tab_layout.addItem(QSpacerItem(0, 0, QMinimum, QExpanding))

    def _init_values(self):
        s = self._app_data.app_settings['openai']

        if s['api_key'] is not None and s['api_key'] != "":
            self.openai_api_key_input.setText(s['api_key'])
            self.openai_sign_up_info.setVisible(False)

        else:
            self.openai_api_key_input.setText("")
            self.openai_sign_up_info.setVisible(True)

        self.openai_auto_max.setChecked(s['auto_max_tokens'])
        self.openai_append_mode.setChecked(s['append_mode'])


        self.openai_temperature_input.setValue(s['temperature'])
        self.openai_max_tokens_input.setValue(s['max_tokens'])
        self.openai_n_completions_input.setValue(s['n_completions'])
        self.openai_join_lines_under_input.setValue(s['join_short_lines'])

        self.openai_max_tokens_input.setDisabled(self.openai_auto_max.isChecked())

    def _save_all_settings(self):
        api_key = self.openai_api_key_input.text()
        if api_key == "":
            api_key = None

        self.openai_max_tokens_input.setDisabled(self.openai_auto_max.isChecked())

        openai_settings = OpenAISettings(
            api_key = api_key,
            model = OpenAIModelToEnum[self.openai_model_select.currentText()],
            temperature = self.openai_temperature_input.value(),
            max_tokens = self.openai_max_tokens_input.value(),
            auto_max_tokens = self.openai_auto_max.isChecked(),
            n_completions = self.openai_n_completions_input.value(),
            join_short_lines = self.openai_join_lines_under_input.value(),
            append_mode = self.openai_append_mode.isChecked(),
        )

        self._app_data.app_settings['openai'] = openai_settings
        self._app_data._save_app_settings()

    def _update_model_max(self):
        model = self._app_data.app_settings['openai']['model']
        model_max = model_max_tokens(model)
        self.openai_max_tokens_input.setMaximum(model_max)
        self.token_warning_msg.setText(f"Warning: max total tokens for {model} is {model_max}")
        self._update_token_count()

    def _append_mode_toggled(self):
        self._update_horiz_splitter_widths()
        self._save_all_settings()

    def _auto_max_toggled(self):
        if self.openai_auto_max.isChecked():
            self._update_token_count()

        self._save_all_settings()

    def _toggle_sidebar(self):
        self.sidebar_visible = not self.sidebar_visible
        self._update_vert_splitter_widths()

    def _update_vert_splitter_widths(self):
        if self.sidebar_visible:
            self.vert_splitter.setSizes([2000, 2000])
        else:
            self.vert_splitter.setSizes([2000, 0])

    def _update_horiz_splitter_widths(self):
        if self.openai_append_mode.isChecked():
            self.horiz_splitter.setSizes([2000, 0])
        else:
            self.horiz_splitter.setSizes([2000, 2000])

    def _show_info(self, text: str):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Information")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setText(text)
        box.exec()

    def _show_warning(self, text: str):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Warning")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setText(text)
        box.exec()

    def _parse_prompt_variables(self,
                                prompt: str,
                                parse_sutta_in_text = False,
                                sutta_uid: Optional[str] = None,
                                selection_text: Optional[str] = None) -> str:

        prompt = prompt.strip()

        if parse_sutta_in_text:
            prompt = self._parse_prompt_sutta_in_text(prompt)

        if sutta_uid:
            prompt = self._parse_prompt_current_sutta(prompt, sutta_uid)

        if selection_text:
            prompt = self._parse_prompt_selection_text(prompt, selection_text)

        return prompt

    def _parse_prompt_selection_text(self, prompt: str, selection_text: str) -> str:
        if '<<selection_text>>' not in prompt:
            return prompt

        return prompt.replace('<<selection_text>>', selection_text)

    def _parse_prompt_current_sutta(self, prompt: str, sutta_uid: str) -> str:
        if '<<current_sutta>>' not in prompt:
            return prompt

        res: List[USutta] = []
        r = self._app_data.db_session \
                            .query(Am.Sutta) \
                            .filter(Am.Sutta.uid == sutta_uid) \
                            .first()
        if r:
            res.append(r)

        r = self._app_data.db_session \
                            .query(Um.Sutta) \
                            .filter(Um.Sutta.uid == sutta_uid) \
                            .first()
        if r:
            res.append(r)

        if len(res) > 0:
            sutta_plain = self._sutta_content_plain(res[0])
            return prompt.replace('<<current_sutta>>', sutta_plain)

        else:
            return prompt

    def _parse_prompt_sutta_in_text(self, prompt: str) -> str:
        matches = re.finditer(r'<<suttas*/([^>]+)>>', prompt)
        parsed_prompt = prompt

        already_replaced = []

        for m in matches:
            if m.group(0) in already_replaced:
                continue

            uid = m.group(1)

            res: List[USutta] = []
            r = self._app_data.db_session \
                              .query(Am.Sutta) \
                              .filter(Am.Sutta.uid == uid) \
                              .first()
            if r:
                res.append(r)

            r = self._app_data.db_session \
                              .query(Um.Sutta) \
                              .filter(Um.Sutta.uid == uid) \
                              .first()
            if r:
                res.append(r)

            if len(res) > 0:
                sutta_plain = self._sutta_content_plain(res[0])
                parsed_prompt = re.sub(m.group(0), sutta_plain, parsed_prompt)
                already_replaced.append(m.group(0))

        return parsed_prompt

    def _sutta_content_plain(self, sutta: USutta) -> str:
        max = self._app_data.app_settings['openai']['join_short_lines']
        content = sutta_content_plain(sutta, max)
        return content

    def _submit_prompt(self):
        openai_settings = self._app_data.app_settings['openai']
        api_key = openai_settings['api_key']

        if api_key is None or api_key == "":
            self._show_warning("<p>Please add your OpenAI key in the Settings tab.</p>")
            return

        messages: List[ChatMessage] = []

        messages.append(
            ChatMessage(
                role=ChatRole.System,
                content=self.system_prompt_input.toPlainText().strip(),
            ))

        text = self.user_prompt_input.toPlainText().strip()
        user_prompt = self._parse_prompt_variables(text, parse_sutta_in_text=True)

        if len(user_prompt) < 4:
            return

        messages.append(
            ChatMessage(
                role=ChatRole.User,
                content=user_prompt,
            ))

        if self.completion_worker is not None:
            self.completion_worker.will_emit_finished = False

        self.completion_worker = CompletionWorker(messages, openai_settings)

        self.completion_worker.signals.finished.connect(partial(self._completion_finished))

        def _completion_error(msg: str):
            if self.completion_worker is not None:
                self.completion_worker.will_emit_finished = False

            self.stop_loading_animation()
            self._show_warning(msg)

        def _completion_warning(msg: str):
            self.completion_warning_msg.setText(msg)

        self.completion_worker.signals.error.connect(partial(_completion_error))
        self.completion_worker.signals.warning.connect(partial(_completion_warning))

        self.start_loading_animation()

        self.thread_pool.start(self.completion_worker)

    def _completion_finished(self, results: List[str]):
        self.stop_loading_animation()
        self.completion_warning_msg.setText("")

        if len(results) == 0:
            return

        # TODO: Add interface elements to show all returned choices.
        # Only using the first returned choice for now.
        result = results[0]

        append_mode = self._app_data.app_settings['openai']['append_mode']

        name_path = self.prompt_name_input.text()
        user_prompt = self.user_prompt_input.toPlainText()

        if append_mode:
            self.user_prompt_input.setPlainText(user_prompt + "\n\n[SYSTEM]:\n" + result + "\n\n[USER]:\n")
            self.completion_text.setPlainText("")

            vert_bar = self.user_prompt_input.verticalScrollBar()
            if vert_bar:
                scroll_bar = self.user_prompt_input.verticalScrollBar()
                if scroll_bar:
                    vert_bar.setValue(scroll_bar.maximum())

        else:
            self.completion_text.setPlainText(result)

        messages: List[ChatMessage] = []

        messages.append(
            ChatMessage(
                role=ChatRole.System,
                content=self.system_prompt_input.toPlainText().strip(),
            ))

        messages.append(
            ChatMessage(
                role=ChatRole.User,
                content=user_prompt,
            ))

        messages.append(
            ChatMessage(
                role=ChatRole.System,
                content=result,
            ))

        log = Um.GptHistory(name_path = name_path, messages_json = json.dumps(messages))

        self._app_data.db_session.add(log)
        self._app_data.db_session.commit()

        self.reload_history_table()

    def _inputs_to_messages(self, parse_variables = False) -> List[ChatMessage]:
        messages: List[ChatMessage] = []

        messages.append(
            ChatMessage(
                role=ChatRole.System,
                content=self.system_prompt_input.toPlainText().strip(),
            ))

        text = self.user_prompt_input.toPlainText().strip()
        if parse_variables:
            text = self._parse_prompt_variables(text)

        messages.append(
            ChatMessage(
                role=ChatRole.User,
                content=text,
            ))

        messages.append(
            ChatMessage(
                role=ChatRole.System,
                content=self.completion_text.toPlainText().strip(),
            ))

        return messages

    def _user_typed(self):
        if not self._input_timer.isActive():
            self._input_timer = QTimer()
            self._input_timer.timeout.connect(partial(self._update_token_count))
            self._input_timer.setSingleShot(True)

        self._input_timer.start(SEARCH_TIMER_SPEED)

    def _tokenizer_finished(self, p: int):
        # p = prompt token count

        auto_max = self._app_data.app_settings['openai']['auto_max_tokens']
        model_max = model_max_tokens(self._app_data.app_settings['openai']['model'])
        if auto_max:
            min_val = self.openai_max_tokens_input.minimum()
            max_val = self.openai_max_tokens_input.maximum()
            m = min(max(model_max - p, min_val), max_val)
            self.openai_max_tokens_input.setValue(m)

        else:
            m = self.openai_max_tokens_input.value()

        total = p+m
        self.token_count_msg.setText(f"{p} (prompt) + {m} = {total} tokens")

        self.token_warning_msg.setVisible(total > model_max)

    def _tokenizer_error(self, msg: str):
        self.token_count_msg.setText(msg)

        auto_max = self._app_data.app_settings['openai']['auto_max_tokens']
        if auto_max:
            self.openai_auto_max.setChecked(False)
            self._app_data.app_settings['openai']['auto_max_tokens'] = False
            self._app_data._save_app_settings()

    def _update_token_count(self):
        if self.tokenizer_worker is not None:
            self.tokenizer_worker.will_emit_finished = False

        messages = self._inputs_to_messages(parse_variables=True)

        model = self._app_data.app_settings['openai']['model']
        self.tokenizer_worker = TokenizerWorker(model, messages)

        self.tokenizer_worker.signals.finished.connect(partial(self._tokenizer_finished))
        self.tokenizer_worker.signals.error.connect(partial(self._tokenizer_error))

        self.thread_pool.start(self.tokenizer_worker)

    def start_loading_animation(self):
        self.completion_loading_bar.setMovie(self._loading_bar_anim)
        self._loading_bar_anim.start()

        icon_processing = QtGui.QIcon()
        icon_processing.addPixmap(QtGui.QPixmap(":/stopwatch"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.prompt_submit.setIcon(icon_processing)

    def stop_loading_animation(self):
        self._loading_bar_anim.stop()
        self.completion_loading_bar.setMovie(self._loading_bar_empty_anim)

        self.prompt_submit.setIcon(QtGui.QIcon())

    def _handle_selection_changed(self, selected: QItemSelection, _: QItemSelection):
        indexes = selected.indexes()
        if len(indexes) > 0:
            self._handle_prompts_tree_clicked(indexes[0])

    def _reset_settings(self):
        self._app_data.app_settings['openai'] = default_openai_settings()
        self._app_data._save_app_settings()
        self._init_values()

    def _prompt_clear_all(self):
        self.prompt_name_input.setText("")
        self.system_prompt_input.setPlainText("")
        self.user_prompt_input.setPlainText("")
        self.completion_text.setPlainText("")
        self.completion_warning_msg.setText("")

    def _prompt_save(self):
        name_path = self.prompt_name_input.text().strip()

        if name_path == "":
            t = datetime.now().strftime("%F %T")
            name_path = f"Prompt {t}"
            self.prompt_name_input.setText(name_path)

        prompt: Optional[Um.GptPrompt] = None

        prompt = self._app_data.db_session \
            .query(Um.GptPrompt) \
            .filter(Um.GptPrompt.name_path == name_path) \
            .first()

        if prompt is None:
            prompt = Um.GptPrompt(
                name_path = name_path,
                messages_json = json.dumps(self._inputs_to_messages()),
                show_in_context = False,
            )
            self._app_data.db_session.add(prompt)

        else:
            prompt.messages_json = json.dumps(self._inputs_to_messages())

        self._app_data.db_session.commit()
        self.reload_prompts_tree()

    def _get_selected_prompt(self) -> Optional[Um.GptPrompt]:
        a = self.prompts_tree_view.selectedIndexes()
        if not a:
            return

        # only one tree node is selected at a time
        idx = a[0]
        item: PromptItem = self.prompts_tree_model.itemFromIndex(idx) # type: ignore

        prompt = self._app_data.db_session \
            .query(Um.GptPrompt) \
            .filter(Um.GptPrompt.id == item.data['db_id']) \
            .first()

        return prompt

    def _prompt_delete_selected(self):
        prompt = self._get_selected_prompt()
        if prompt:
            self._app_data.db_session.delete(prompt)
            self._app_data.db_session.commit()

            self.reload_prompts_tree()

    def _prompt_delete_all(self):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Delete Confirmation")
        box.setText("<p>Delete all Prompts?</p>")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        reply = box.exec()
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._app_data.db_session.query(Um.GptPrompt).delete()
        self._app_data.db_session.commit()

        self.reload_prompts_tree()

    def _prompt_copy(self):
        user_prompt_text = self.user_prompt_input.toPlainText().strip()
        if user_prompt_text != "":
            self._app_data.clipboard_setText(user_prompt_text)

    def _prompt_copy_completion(self):
        completion_text = self.completion_text.toPlainText().strip()
        if completion_text != "":
            self._app_data.clipboard_setText(completion_text)

    def _prompt_copy_all(self):
        prompt_name = self.prompt_name_input.text().strip()
        system_prompt_text = self.user_prompt_input.toPlainText().strip()
        user_prompt_text = self.user_prompt_input.toPlainText().strip()
        completion_text = self.completion_text.toPlainText().strip()

        if system_prompt_text != "" or user_prompt_text != "" or completion_text != "":
            all_text = f"{prompt_name}\n\n{system_prompt_text}\n\n{user_prompt_text}\n\n{completion_text}".strip()
            self._app_data.clipboard_setText(all_text)

    def _prompt_toggle_menu(self):
        prompt = self._get_selected_prompt()
        if not prompt:
            return

        if prompt.show_in_context is None or not prompt.show_in_context:
            prompt.show_in_context = True
        else:
            prompt.show_in_context = False

        self._app_data.db_session.commit()

        a = self.prompts_tree_view.selectedIndexes()
        idx = a[0]
        sel_row = idx.row()

        self.reload_prompts_tree()

        idx = self.prompts_tree_model.index(sel_row, 0)
        sel_model = self.prompts_tree_view.selectionModel()
        if sel_model:
            sel_model.select(idx,
                             QItemSelectionModel.SelectionFlag.ClearAndSelect | \
                             QItemSelectionModel.SelectionFlag.Rows)

    def _prompt_show_parsed(self):
        prompt_name = self.prompt_name_input.text().strip()
        system_prompt_text = self.system_prompt_input.toPlainText().strip()
        text = self.user_prompt_input.toPlainText().strip()
        user_prompt_text = self._parse_prompt_variables(text, parse_sutta_in_text=True)
        completion_text = self.completion_text.toPlainText().strip()

        text = f"{prompt_name}\n\n{system_prompt_text}\n\n{user_prompt_text}\n\n{completion_text}".strip()

        d = ShowPromptDialog(text)
        d.exec()

    def _handle_history_row_load(self):
        a = self.history_table.selectedIndexes()
        if len(a) != 0:
            self._handle_history_load(a[0])

    def _handle_history_load(self, val: QModelIndex):
        model = val.model()
        if model is None:
            return

        data = model.data(val, Qt.ItemDataRole.UserRole)
        db_id = int(data[val.row()][HistoryModelColToIdx['_db_id']])

        res = self._app_data.db_session \
            .query(Um.GptHistory) \
            .filter(Um.GptHistory.id == db_id) \
            .first()

        if not res:
            return

        self.prompt_name_input.setText("" if res.name_path is None else res.name_path)

        if res.messages_json is None:
            messages = []
        else:
            messages = json.loads(res.messages_json)

        self._set_prompt_inputs(messages)

    def _history_delete_selected(self):
        a = self.history_table.selectedIndexes()
        if not a:
            return

        db_ids = set(map(lambda idx: self.history_model._data[idx.row()][HistoryModelColToIdx['_db_id']], a))

        n = len(db_ids)
        if n > 1:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Delete Confirmation")
            box.setText(f"<p>Delete {n} GPT history entries?</p>")
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            reply = box.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return

        items = self._app_data.db_session \
                              .query(Um.GptHistory) \
                              .filter(Um.GptHistory.id.in_(db_ids)) \
                              .all()

        for i in items:
            self._app_data.db_session.delete(i)

        self._app_data.db_session.commit()

        self.reload_history_table()

    def _history_delete_all(self):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Delete Confirmation")
        box.setText("<p>Delete all GPT history entries?</p>")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        reply = box.exec()
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._app_data.db_session.query(Um.GptHistory).delete()
        self._app_data.db_session.commit()

        self.reload_history_table()

    def _handle_import(self):
        file_path, _ = QFileDialog \
            .getOpenFileName(self,
                            "Import from CSV...",
                            "",
                            "CSV Files (*.csv)")

        if len(file_path) == 0:
            return

        n = self._app_data.import_prompts(file_path)

        self.reload_history_table()
        self.reload_prompts_tree()

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"Imported {n} prompts.")
        box.setWindowTitle("Import Completed")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _handle_export(self):
        file_path, _ = QFileDialog \
            .getSaveFileName(self,
                             "Export as CSV...",
                             "",
                             "CSV Files (*.csv)")

        if len(file_path) == 0:
            return

        n = self._app_data.export_prompts(file_path)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(f"Exported {n} prompts.")
        box.setWindowTitle("Export Completed")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _handle_close(self):
        self.close()

    def _connect_signals(self):
        self.action_Close_Window \
            .triggered.connect(partial(self._handle_close))

        self.action_Import \
            .triggered.connect(partial(self._handle_import))

        self.action_Export \
            .triggered.connect(partial(self._handle_export))

        self.toggle_sidebar_btn.clicked.connect(partial(self._toggle_sidebar))

        sel_model = self.prompts_tree_view.selectionModel()
        if sel_model:
            sel_model.selectionChanged.connect(partial(self._handle_selection_changed))

        self.prompt_submit.clicked.connect(partial(self._submit_prompt))

        self.prompt_clear_all_btn.clicked.connect(partial(self._prompt_clear_all))
        self.prompt_save_btn.clicked.connect(partial(self._prompt_save))
        self.prompt_show_parsed_btn.clicked.connect(partial(self._prompt_show_parsed))
        self.prompt_toggle_menu_btn.clicked.connect(partial(self._prompt_toggle_menu))

        self.prompt_copy_btn.clicked.connect(partial(self._prompt_copy))
        self.prompt_copy_completion_btn.clicked.connect(partial(self._prompt_copy_completion))
        self.prompt_copy_all_btn.clicked.connect(partial(self._prompt_copy_all))

        self.prompt_delete_btn.clicked.connect(partial(self._prompt_delete_selected))
        self.prompt_delete_all_btn.clicked.connect(partial(self._prompt_delete_all))

        self.history_table.doubleClicked.connect(self._handle_history_load)
        self.history_load_btn.clicked.connect(partial(self._handle_history_row_load))
        self.history_delete_btn.clicked.connect(partial(self._history_delete_selected))
        self.history_delete_all_btn.clicked.connect(partial(self._history_delete_all))

        self.settings_reset_btn.clicked.connect(partial(self._reset_settings))

        self.system_prompt_input.textChanged.connect(partial(self._user_typed))
        self.user_prompt_input.textChanged.connect(partial(self._user_typed))

        self.openai_model_select.currentIndexChanged.connect(partial(self._save_all_settings))
        self.openai_model_select.currentIndexChanged.connect(partial(self._update_model_max))

        self.openai_append_mode.toggled.connect(partial(self._append_mode_toggled))
        self.openai_auto_max.toggled.connect(partial(self._auto_max_toggled))

        for i in [self.openai_temperature_input,
                  self.openai_max_tokens_input,
                  self.openai_n_completions_input,
                  self.openai_join_lines_under_input]:
            i.valueChanged.connect(self._save_all_settings)

        for i in [self.openai_api_key_input,]:
            i.textChanged.connect(self._save_all_settings)

class CompletionWorkerSignals(QObject):
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    finished = pyqtSignal(list)

class CompletionWorker(QRunnable):
    signals: CompletionWorkerSignals

    def __init__(self, messages: List[ChatMessage], openai_settings: OpenAISettings):
        super().__init__()

        api_key = openai_settings['api_key']
        if api_key is None or api_key == "":
            logger.error("OpenAI API key is None")
            return

        import openai
        self.openai = openai

        self.openai.api_key = openai_settings['api_key']

        self.signals = CompletionWorkerSignals()
        self.messages = messages
        self.openai_settings = openai_settings

        self.query_started_time: datetime = datetime.now()
        self.query_finished_time: Optional[datetime] = None

        self.will_emit_finished = True

    @pyqtSlot()
    def run(self):
        logger.info("CompletionWorker::run()")
        try:
            results = self.chat_completion()

            if self.will_emit_finished:
                logger.info("CompletionWorker::run() signals.finished.emit()")
                self.signals.finished.emit(results)

        except Exception as e:
            logger.error(e)
            self.signals.error.emit(f"<p>OpenAI Completion error:</p><p>{e}</p>")

    def chat_completion(self, max_retries = 10) -> List[str]:
        logger.info("gpt3_chat()")

        content: List[str] = []
        try_count = 1
        while try_count <= max_retries:
            try:
                logger.info(f"Request ChatCompletion, try_count {try_count}")

                # https://platform.openai.com/docs/api-reference/chat/create
                resp: ChatResponse = self.openai.ChatCompletion.create( # type: ignore
                    model =  OpenAIModelLatest[self.openai_settings['model']],
                    messages = self.messages,
                    temperature = self.openai_settings['temperature'],
                    max_tokens = self.openai_settings['max_tokens'],
                    n = self.openai_settings['n_completions'],
                    stream = False,
                )

                content = [i['message']['content'].strip() for i in resp['choices']]
                break

            except Exception as e:
                logger.error(f"---\n{e}\n---")
                # The model: `gpt-4` does not exist
                if "does not exist" in str(e):
                    raise Exception(f"ChatGPT request failed. Error: {e}")

                if "maximum context length" in str(e):
                    # Return the exception text as a response, so the script can continue.
                    return [str(e)]

                msg = f"ChatGPT Request failed, retrying ({try_count})."
                self.signals.warning.emit(msg)
                logger.error(msg)

            try_count += 1

        if len(content) > 0:
            return content
        else:
            raise Exception(f"ChatGPT request failed, max_retries {max_retries} reached.")

class TokenizerWorkerSignals(QObject):
    error = pyqtSignal(str)
    finished = pyqtSignal(int)

class TokenizerWorker(QRunnable):
    signals: TokenizerWorkerSignals

    def __init__(self,
                 model: OpenAIModel,
                 messages: List[ChatMessage]):

        super().__init__()

        self.signals = TokenizerWorkerSignals()

        self.model = model
        self.messages = messages

        self.tokenizer: Optional[Any] = None

        self.will_emit_finished = True

    @pyqtSlot()
    def run(self):
        try:
            count = num_tokens_from_messages(self.messages, model=OpenAIModelLatest[self.model])
            if self.will_emit_finished:
                self.signals.finished.emit(count)

        except Exception as e:
            msg = f"Tokenizer error: {e}"
            logger.error(msg)
            self.signals.error.emit(msg)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """
    Returns the number of tokens used by a list of messages.

    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """

    try:
        encoding = tiktoken.encoding_for_model(model)

    except KeyError:
        logger.warn("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:

        tokens_per_message = 3
        tokens_per_name = 1

    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted

    elif "gpt-3.5-turbo" in model:
        # logger.info("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")

    elif "gpt-4" in model:
        # logger.info("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")

    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    return num_tokens

def messages_concat(messages: List[ChatMessage]) -> str:
    prompt = "\n\n".join([i["content"] for i in messages])
    return prompt
