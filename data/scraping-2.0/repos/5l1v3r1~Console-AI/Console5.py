
import signal
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QApplication, QSizePolicy, QMainWindow, QDialog, QGridLayout, QFrame, QSpinBox
from PyQt5.QtGui import QFont, QPixmap
import openai
import requests
import os, sys, platform
is_windows = True if platform.system() == "Windows" else False

if is_windows:
    os.system("title Mizogg @ github.com/Mizogg")

def water(text):
    os.system(""); faded = ""
    green = 10
    for line in text.splitlines():
        faded += (f"\033[38;2;0;{green};255m{line}\033[0m\n")
        if not green == 255:
            green += 15
            if green > 255:
                green = 255
    return faded

def red(text):
    os.system(""); faded = ""
    for line in text.splitlines():
        green = 250
        for character in line:
            green -= 5
            if green < 0:
                green = 0
            faded += (f"\033[38;2;255;{green};0m{character}\033[0m")
        faded += "\n"
    return faded
    
class CommandThread(QThread):
    commandOutput = pyqtSignal(str)
    commandFinished = pyqtSignal(int)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        if self.command.startswith('python'):
            script = self.command[7:].strip()
            if script.endswith('.py'):
                self.process = subprocess.Popen(
                    self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                    universal_newlines=True
                )
                for line in self.process.stdout:
                    output = line.strip()
                    self.commandOutput.emit(output)
                self.process.stdout.close()
                self.commandFinished.emit(self.process.wait())
            else:
                try:
                    result = eval(self.command[7:])
                    self.commandOutput.emit(str(result))
                    self.commandFinished.emit(0)
                except Exception as e:
                    self.commandOutput.emit(str(e))
                    self.commandFinished.emit(1)
        else:
            self.process = subprocess.Popen(
                self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                universal_newlines=True
            )
            for line in self.process.stdout:
                output = line.strip()
                self.commandOutput.emit(output)
            self.process.stdout.close()
            self.commandFinished.emit(self.process.wait())

class ConsoleWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.consoleOutput = QPlainTextEdit(self)
        self.consoleOutput.setReadOnly(True)
        self.consoleOutput.setFont(QFont("Courier"))
        self.layout.addWidget(self.consoleOutput)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.consoleOutput.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @pyqtSlot(int)
    def update_console_style(self, index):
        if index == 1:
            self.consoleOutput.setStyleSheet("background-color: white; color: purple; font-size: 14px;")
        elif index == 2:
            self.consoleOutput.setStyleSheet("background-color: black; color: green; font-size: 14px;")
        elif index == 3:
            self.consoleOutput.setStyleSheet("background-color: red; color: blue; font-size: 14px;")

    @pyqtSlot(str)
    def append_output(self, output):
        self.consoleOutput.appendPlainText(output)

class ImageViewerDialog(QDialog):
    def __init__(self, image_urls, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.layout = QGridLayout(self)

        row = 0
        col = 0
        for image_url in image_urls:
            frame = QFrame(self)
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setLineWidth(1)
            frame_layout = QVBoxLayout(frame)
            image_label = QLabel(self)
            frame_layout.addWidget(image_label)

            response = requests.get(image_url)
            pixmap = QPixmap()
            pixmap.loadFromData(response.content)
            image_label.setPixmap(pixmap.scaledToWidth(1000))
            image_label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(frame, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        self.setMinimumSize(900, 500)

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Console.py")
        self.setGeometry(100, 100, 780, 560)
        self.process = None
        self.commandThread = None
        self.scanning = False
        self.initUI()
        openai.api_key = '' #Your API KEY

    def initUI(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.layout = QVBoxLayout(self)
        self.consoleWindow = ConsoleWindow(self)
        self.layout.addWidget(self.consoleWindow)

        self.customLayout = QHBoxLayout()
        self.layout.addLayout(self.customLayout)
        self.customLabel = QLabel("Custom CMD here:", self)
        self.customLayout.addWidget(self.customLabel)
        self.inputcustomEdit = QLineEdit(self)
        self.inputcustomEdit.setPlaceholderText('Type command')
        self.inputcustomEdit.returnPressed.connect(self.custom_start)
        self.customLayout.addWidget(self.inputcustomEdit)

        self.customButton = QPushButton("Execute", self)
        self.customButton.setStyleSheet("color: green;")
        self.customButton.clicked.connect(self.custom_start)
        self.customLayout.addWidget(self.customButton)
        self.stopButton = QPushButton("Stop", self)
        self.stopButton.setStyleSheet("color: red;")
        self.stopButton.clicked.connect(self.stop_exe)
        self.customLayout.addWidget(self.stopButton)

        self.aiButton = QPushButton("AI Interaction", self)
        self.aiButton.setStyleSheet("color: blue;")
        self.aiButton.clicked.connect(self.ai_interaction)
        self.customLayout.addWidget(self.aiButton)

        self.aiPictureLayout = QHBoxLayout()
        self.aiPictureButton = QPushButton("AI Picture", self)
        self.aiPictureButton.setStyleSheet("color: orange;")
        self.aiPictureButton.clicked.connect(self.ai_picture)
        self.aiPictureLayout.addWidget(self.aiPictureButton)
        self.pictureSpinBox = QSpinBox(self)
        self.pictureSpinBox.setMinimum(1)
        self.pictureSpinBox.setValue(1)
        self.aiPictureLayout.addWidget(self.pictureSpinBox)
        self.customLayout.addLayout(self.aiPictureLayout)

        self.colourLayout = QHBoxLayout()
        self.colorlable = QLabel(
            '<html><b><font color="blue" size="3"> Pick Console Colour </font></b></html>', self
        )
        self.colorlable.setAlignment(Qt.AlignRight)
        self.colorComboBox = QComboBox(self)
        self.colorComboBox.addItem("Pick Console Colour")
        self.colorComboBox.addItem("Option 1: White Background, Purple Text")
        self.colorComboBox.addItem("Option 2: Black Background, Green Text")
        self.colorComboBox.addItem("Option 3: Red Background, Blue Text")
        self.colorComboBox.currentIndexChanged.connect(self.update_console_style)
        self.colourLayout.addWidget(self.colorlable)
        self.colourLayout.addWidget(self.colorComboBox)
        self.layout.addLayout(self.colourLayout)

        self.layout.setContentsMargins(0, 0, 0, 0)

    def custom_start(self):
        command = self.inputcustomEdit.text().strip()
        self.execute_command(command)

    def execute_command(self, command):
        if self.scanning:
            return

        self.scanning = True

        if self.commandThread and self.commandThread.isRunning():
            self.commandThread.terminate()
        self.consoleWindow.append_output(f"> {command}")
        self.consoleWindow.append_output("")
        self.commandThread = CommandThread(command)
        self.commandThread.commandOutput.connect(self.append_output)
        self.commandThread.commandFinished.connect(self.command_finished)
        self.commandThread.start()
        self.timer.start(100)

    def stop_exe(self):
        if self.commandThread and self.commandThread.isRunning():
            if platform.system() == "Windows":
                subprocess.Popen(["taskkill", "/F", "/T", "/PID", str(self.commandThread.process.pid)])
            else:
                os.killpg(os.getpgid(self.commandThread.process.pid), signal.SIGTERM)

            self.timer.stop()
            self.scanning = False
            returncode = 'Closed'
            self.command_finished(returncode)

    @pyqtSlot(str)
    def append_output(self, output):
        self.consoleWindow.append_output(output)

    @pyqtSlot(int)
    def command_finished(self, returncode):
        self.timer.stop()
        self.scanning = False

        if returncode == 0:
            self.append_output("Command execution finished successfully")
        elif returncode == 'Closed':
            self.append_output("Process has been stopped by the user")
        else:
            self.append_output("Command execution failed")

    @pyqtSlot(int)
    def update_console_style(self, index):
        self.consoleWindow.update_console_style(index)

    def ai_interaction(self):
        question = self.inputcustomEdit.text().strip()
        self.consoleWindow.append_output(f"> {question}")
        self.consoleWindow.append_output("")
        ai_response = self.complete_chat(question)
        self.consoleWindow.append_output(ai_response)
        self.consoleWindow.append_output("")

    def ai_picture(self):
        question = self.inputcustomEdit.text().strip()
        self.consoleWindow.append_output(f"> {question}")
        self.consoleWindow.append_output("")
        ai_response = self.complete_chat(question)
        n = self.pictureSpinBox.value()
        res = openai.Image.create(prompt=question, n=n, size="1024x1024")
        image_urls = [item['url'] for item in res["data"]]

        dialog = ImageViewerDialog(image_urls, self)
        dialog.exec_()

    def complete_chat(self, prompt):
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7,
            n=1,
            stop=None,
        )
        return response.choices[0].text.strip()

    def update_gui(self):
        QApplication.processEvents()


if __name__ == '__main__':
    mizogg= f'''
                      ___            ___
                     (o o)          (o o)
                    (  V  ) MIZOGG (  V  )
                    --m-m------------m-m--
                  © mizogg.co.uk 2018 - 2023
                        
                        Console.py 

                    VIP PROJECT Mizogg
                 
    {red(f"[>] Running with Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")}


'''
    print(water(mizogg), end="")
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
