import importlib
import os
import random
import sys
import time
import openai

import pyautogui  # Import pyautogui for mouse control
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon, QTextCharFormat, QColor, QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QWidget, QVBoxLayout, QCheckBox, QPushButton, \
    QLabel, QLineEdit, QTextEdit, QHBoxLayout, QHeaderView, QSizePolicy

from libs.ai import bot
from libs.capture import capture_screen, ocr_utils
from libs.delay import NonBlockingDelay
from libs.image import imageManipulator
from libs.macro import search_macros, Macro, TerminatedError
from libs.peon import Peon
from libs.sprite import load_sprites
from ui import chatLogWindow
from ui.mainWindow import Ui_MainWindow
from ui.chatLogWindow import Ui_ChatLogWindow

class AutoHyakuretsu(QMainWindow, Ui_MainWindow):
    macros: dict[str, Macro] = []
    sprites: dict = {}
    image_manipulator: imageManipulator = None
    ocr_utils: ocr_utils = None
    bot: bot = None
    chat_window: QMainWindow = None
    chat_log_ui: Ui_ChatLogWindow = None
    peon: Peon = None

    def __init__(self):
        super().__init__()
        self.timer = None
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.macros_dir = self.script_directory + "/macros"
        self.sounds_dir = self.script_directory + "/sounds"
        self.macros = {}
        self.running = False
        self.timer_interval = 10
        self.loop_rate = 5000
        self.loop_countdown = 0
        self.sound = False
        self.lol = False
        self.filter_line_edits = []
        self.filter = {}
        self.ui_form = Ui_MainWindow()
        self.image_manipulator = imageManipulator()
        self.ocr_utils = ocr_utils(self)
        self.bot = bot(self.handle_bot_response)
        #self.bot.autogen_test_local()
        #self.bot.autogen_test()
        self.peon = Peon()
        self.init_ui()
        self.update_ui_from_model()
        self.load_data()

    def init_ui(self):
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        form = self.ui_form

        form.setupUi(self)
        form.soundCheckbox.stateChanged.connect(self.update_model_from_ui)
        form.lolCheckbox.stateChanged.connect(self.update_model_from_ui)
        form.toggleButton.clicked.connect(self.toggle_action)
        form.pollCountdown.setMinimum(0)
        form.pollRate.textChanged.connect(self.update_model_from_ui)

        table = form.macrosTable
        table.setColumnCount(5)
        header_labels = ["active", "Module", "Macro", "run_nr", "run"]
        table.setHorizontalHeaderLabels(header_labels)
        header = table.horizontalHeader()
        #for col in range(table.columnCount()):
        #    header.setSectionResizeMode(col, QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)


        table.setSortingEnabled(True)

        # create a horizontal layout for the labels and line edits
        layout = QHBoxLayout(self.ui_form.centralwidget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        for col, label in enumerate(["Module", "Macro"]):
            # create a label and add it to the layout
            label_widget = QLabel(label)
            layout.addWidget(label_widget)
            # create a line edit and add it to the layout
            line_edit = QLineEdit(self)
            line_edit.setPlaceholderText(f"Filter {label}")
            # set self.filter[label] to the line edit, then call self.filter_table
            line_edit.textChanged.connect(lambda text, label=label: self.filter.update({label: text}))
            # add a delay before filtering the table
            line_edit.textChanged.connect(lambda: QTimer.singleShot(10, self.filter_table))
            layout.addWidget(line_edit)
            # add a spacer to the layout
            spacer = QWidget(self)
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            layout.addWidget(spacer)


        # add the layout to the main layout
        self.ui_form.verticalLayout.addLayout(layout)

        table.sortByColumn(0, Qt.AscendingOrder)  # Sort column 0 in ascending order
        # main loop timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timer_action)
        self.timer.start(self.timer_interval)

        self.toggle_chat_window()

        self.refresh_ui()

    def filter_table(self):
        print(self.filter)
        for row in range(self.ui_form.macrosTable.rowCount()):
            # get the row data as a dict
            row_data = self.get_row_data_as_dict(self.ui_form.macrosTable, row)
            # cycle self.filter
            for key, value in self.filter.items():
                # omit empty filter fields
                if not value:
                    continue
                # if the value is not in the row data, hide the row
                if value.lower() not in row_data[key].lower():
                    self.ui_form.macrosTable.hideRow(row)
                    break
            else:
                # if all values are in the row data, show the row
                self.ui_form.macrosTable.showRow(row)

    def get_row_data_as_dict(self, table_widget, row_number):
        row_data = {}
        for col in range(table_widget.columnCount()):
            item = table_widget.item(row_number, col)
            if item is not None:
                row_data[table_widget.horizontalHeaderItem(col).text()] = item.text()
        return row_data

    def load_data(self):
        self.sprites = load_sprites(self.script_directory + "/sprites", self.sounds_dir)

        macros = search_macros(self.macros_dir)

        table = self.ui_form.macrosTable
        table.setRowCount(len(macros))

        self.macros = {}

        row = 0
        for macro in macros:
            checkbox_widget = CheckBoxCellWidget()
            table.setCellWidget(row, 0, checkbox_widget)

            module_name = macro["module"].replace(self.macros_dir + os.path.sep, "")
            table.setItem(row, 1, QTableWidgetItem(module_name))
            table.setItem(row, 2, QTableWidgetItem(macro["name"]))
            table.setItem(row, 3, QTableWidgetItem(str(macro["run_count"] if "run_count" in macro else 0)))

            macro_name = macro['name']
            spec = importlib.util.spec_from_file_location(macro_name, os.path.join(macro['module'], macro_name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate the class
            if hasattr(module, macro_name):
                class_ = getattr(module, macro_name)
                instance = class_(self, macro_name, os.path.join(self.macros_dir, module_name, 'templates'))
                print(f"Instantiated {macro_name}: {instance.name}")
                self.macros[module_name + '.' + macro_name] = instance

                # Create StartStopButton for start/stop
                button = StartStopButton(instance, self)

                # Create custom item and set it as the cell widget
                table.setCellWidget(row, 4, button)

            row += 1


        screenshot_cv = capture_screen()
        if False:
            address = self.macros['aplan.process_email'].run(screenshot_cv, {
                "prompt" : """
                Buongiorno Alessandro,
    
    Ho aggiunto in copia il collega Mario Baricco, Vostro referente per il merceologico richiesto.
    Sarà compito suo darle seguito alla gentile richiesta
    
    Rimango a disposizione
    A presto
    
    Lorenzo Fasolini 
    
    From: Alessandro Candriello <acandriello@pesstech.com> 
    Sent: Tuesday, October 10, 2023 11:57 AM
    To: Fasolini, Lorenzo <Lorenzo.Fasolini@we-online.com>
    Subject: DipSwitch & Trimmer 
    
    CAUTION: External Mail!
    Buongiorno Lorenzo,
    
    Le chiedo cortesia di quotarmi e fornirmi i tempi di consegna per:
    
    Dipswitch SMD passo 1.27mm 5 posizioni  416131160805                               Pz 1000
    Dipswitch SMD passo 1.27mm 6 posizioni  416131160806                               Pz 1000
    
    Resto in attesa di suo riscontro, grazie.
    
    Best Regards
    Alessandro Candriello
    
    PESS TECHNOLOGIES SRL
    
    Roma
    Via di Grotta Perfetta, 367
    00142 Roma
     
    Pess Tech Srl - Asti
    Via Antica Dogana, 7
    14100 Fraz. Quarto Inferiore - AT
    T. 0141 293821
    VAT IT 01510920059
                """,
                "notes" : ""
            })

    def get_row_from_button(self, button):
        table = self.centralWidget().layout().itemAt(0).widget()
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                widget = table.cellWidget(row, col)
                if isinstance(widget, StartStopButton) and widget == button:
                    return row
        return -1

    def update_model_from_ui(self):
        self.loop_rate = int(self.ui_form.pollRate.text())
        self.sound = self.ui_form.soundCheckbox.isChecked()
        self.lol = self.ui_form.lolCheckbox.isChecked()
        self.refresh_ui()

    def update_ui_from_model(self):
        self.ui_form.pollRate.setValue(self.loop_rate)
        self.ui_form.soundCheckbox.setChecked(self.sound)
        self.ui_form.lolCheckbox.setChecked(self.lol)
        self.refresh_ui()

    def toggle_action(self):
        self.running = not self.running
        self.refresh_ui()
        print("Starting" if self.running else "Stopping")

    def refresh_ui(self):
        # set the label of the button according to running status
        self.ui_form.toggleButton.setText("Stop" if self.running else "Loop")
        # set the progress bar to the next poll
        self.ui_form.pollCountdown.setMaximum(self.loop_rate)
        self.ui_form.pollCountdown.setValue(self.loop_countdown)

    def timer_action(self):
        if not self.running:
            return

        if self.loop_countdown <= 0:
            self.loop_countdown = self.loop_rate
            print(f"Loop triggered at {time.strftime('%H:%M:%S', time.localtime())}")

            selected_rows = self.get_selected_macro_rows()
            if len(selected_rows) > 0:
                screenshot_cv = capture_screen()

                for row in selected_rows:
                    # get the field 'name' from the row
                    macro_module = self.ui_form.macrosTable.item(row, 1).text()
                    macro_name = self.ui_form.macrosTable.item(row, 2).text()
                    macro = self.macros[macro_module + '.' + macro_name]
                    print(f"Running macro {macro_module}.{macro_name}")
                    macro.run(screenshot_cv)

        self.loop_countdown -= self.timer_interval
        self.refresh_ui()

    def click(self, x, y, nr_clicks=1):
        #mouse_x, mouse_y = pyautogui.position()

        for i in range(nr_clicks):
            pyautogui.click(x, y)
            time.sleep(0.1)

        #pyautogui.moveTo(mouse_x, mouse_y)

        print(f"Clicked at {x}, {y}")

        if self.lol:
            sprite = random.choice(list(self.sprites.values()))
            sprite.display_at(x, y, sprite.duration, self.sound)

    def get_selected_macro_rows(self):
        selected_rows = []
        for row in range(self.ui_form.macrosTable.rowCount()):
            checkbox_item = self.ui_form.macrosTable.cellWidget(row, 0)
            if checkbox_item.checkbox.isChecked():
                selected_rows.append(row)

        return selected_rows

    def run_macro_single(self, macro):
        window = QMainWindow()
        window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)

        window.resize(window.screen().size() * 0.3)

        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        label = QLabel("Enter values for required parameters:")
        layout.addWidget(label)

        parameters = macro.get_required_parameters()
        if len(parameters) == 0:
            try:
                macro.run(capture_screen(), {})
            except TerminatedError as te:
                print(f"Terminated: {te}")

            window.close()
            return

        input_fields = {}

        for param_name in parameters:
            paramLayout = QHBoxLayout()
            # the first column will contain labels and must be fixed width at 200px
            label = QLabel(param_name)

            fixedWidthWidget = QWidget()
            fixedWidthWidget.setFixedWidth(100)

            fixedWidthLayout = QHBoxLayout()
            fixedWidthLayout.addWidget(label)

            fixedWidthWidget.setLayout(fixedWidthLayout)

            paramLayout.addWidget(fixedWidthWidget)

            if param_name == 'prompt':
                input_field = QTextEdit()
            else:
                input_field = QLineEdit()

            paramLayout.addWidget(input_field)
            layout.addLayout(paramLayout)
            input_fields[param_name] = input_field

        # add a stretchable space to the bottom of the layout
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(spacer)

        submit_button = QPushButton("Run Macro")
        cancel_button = QPushButton("Cancel")

        button_layout = QHBoxLayout()
        button_layout.addWidget(cancel_button)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button_layout.addWidget(spacer)
        button_layout.addWidget(submit_button)
        layout.addLayout(button_layout)

        def clicked():
            # create a dictionary of input fields values as returned by their text() method
            # text() may not exits, so use toPlainText() for QTextEdit
            values = {k: v.toPlainText() if hasattr(v, 'toPlainText') else v.text() for k, v in input_fields.items()}

            macro.run(capture_screen(), values)
            window.close()

        submit_button.clicked.connect(lambda: clicked())

        def cancel():
            window.close()

        cancel_button.clicked.connect(lambda: cancel())

        window.show()

    def toggle_chat_window(self):

        if self.chat_window:
            self.chat_window.close()
            self.chat_window = None
            return

        self.chat_window = QMainWindow()
        self.chat_window.setWindowFlags(self.chat_window.windowFlags() | Qt.WindowStaysOnTopHint)
        self.chat_window.resize(self.chat_window.screen().size() * 0.5)

        # Create an instance of Ui_ChatLogWindow
        self.chat_log_ui = Ui_ChatLogWindow()

        # Call the setupUi method of chat_log_ui to set up the user interface
        self.chat_log_ui.setupUi(self.chat_window)

        # Set the window title
        self.chat_window.setWindowTitle("Chat Log")

        # Show the window
        self.chat_window.show()

    def print_colored_text(self, text, color='yellow'):
        if self.chat_window is None:
            return

        cursor = self.chat_log_ui.chatLog.textCursor()
        format = QTextCharFormat()
        # Set the text color
        format.setForeground(QColor(color))
        # Move the cursor to the end of the document
        cursor.movePosition(QTextCursor.End)
        # Apply the format to the cursor
        cursor.setCharFormat(format)
        # Insert the text with the applied format
        cursor.insertText(text + '\n')

    def handle_bot_response(self, chat_log):
        last_log = chat_log[-1]
        if 'question' in last_log:
            self.print_colored_text(last_log['question'], 'yellow')
        if 'answer' in last_log:
            self.print_colored_text(last_log['answer'], 'red')

    def closeEvent(self, event):
        # Close all child windows when the main window is closed

        if self.chat_window:
            self.chat_window.close()

        event.accept()

class CheckBoxCellWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.checkbox = QCheckBox()
        self.layout.addWidget(self.checkbox)
        self.setLayout(self.layout)


class StartStopButton(QPushButton):
    macro: Macro = None

    # some constants
    START = 0
    STOP = 1

    def __init__(self, macro: Macro, app):
        super().__init__()
        self.setIcon(QIcon.fromTheme('media-playback-start'))  # Built-in start icon
        self.state = 0  # 0 indicates "Start"
        self.macro = macro
        self.clicked.connect(self.toggle_state)
        self.app = app

    def toggle_state(self):
        if self.state == 0:
            self.setIcon(QIcon.fromTheme('media-playback-stop'))  # Built-in stop icon
            # set disabled to prevent double-clicking
            self.setDisabled(True)
            NonBlockingDelay.wait(100)
            self.state = 1  # 1 indicates "Running"
            self.app.run_macro_single(self.macro)
            self.toggle_state()
        else:
            self.setIcon(QIcon.fromTheme('media-playback-start'))  # Built-in start icon
            self.state = 0  # 0 indicates "Start"
            self.setDisabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = AutoHyakuretsu()

    window.show()
    sys.exit(app.exec_())
