import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from openai import OpenAI
import re

################################### 단축기 지정은 디자이너에서 설정 #############################
form_window = uic.loadUiType('./qt_notepad.ui')[0]

class Exam(QMainWindow, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.edited_flag = False
        self.path = ['제목 없음', '']
        self.title = self.path[0] + " - 힘내 일기"
        self.setWindowTitle(self.title)

        self.actionNew.triggered.connect(self.action_new_slot)
        self.actionOpen.triggered.connect(self.action_open_slot)
        self.actionSave.triggered.connect(self.action_save_slot)
        self.actionSave_as.triggered.connect(self.action_save_as_slot)
        self.actionExit.triggered.connect(self.action_exit_slot)

        self.actionUndo.triggered.connect(self.plainTextEdit.undo)      # 기본으로 제공되는 함수
        self.actionCut.triggered.connect(self.plainTextEdit.cut)        # 기본으로 제공되는 함수
        self.actionCopy.triggered.connect(self.plainTextEdit.copy)      # 기본으로 제공되는 함수
        self.actionPaste.triggered.connect(self.plainTextEdit.paste)    # 기본으로 제공되는 함수
        self.actionDelete.triggered.connect(self.plainTextEdit.cut)     # delete는 제공되지 않음
        self.actionFont.triggered.connect(self.action_font_slot)
        self.actionAbout.triggered.connect(self.action_about_slot)
        self.plainTextEdit.textChanged.connect(self.text_changed_slot)  # plainTextEdit에 입력이 들어오는 지 확인
        self.statusbar.showMessage(self.path[0])        # statusbar 에 메시지 띄우는 방법
        self.btn.clicked.connect(self.btn_clear_clicked_slot)
        self.comboBox.currentIndexChanged.connect(self.combobox_slot)

    def btn_clear_clicked_slot(self):
        text = self.plainTextEdit.toPlainText()
        chatbot = self.combobox_slot()
        if chatbot==0:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    # {"role": "system", "content": "당신은 반말을 사용하는 인공지능 챗봇입니다. 입력된 일기를 읽고, 친구에게 써주듯이 코멘트를 작성해주세요."},
                    {"role": "system", "content": "당신은 친절한 인공지능 챗봇입니다. 입력된 일기를 읽고,상담사가 상담자에게 상담해주듯이 코멘트를 친절하게 100자 이내로 작성해주세요."},
                    # {"role": "system", "content": "당신은 sns어체를 사용하는 인공지능 챗봇입니다. 입력된 일기를 읽고, 실제 사람들이 쓰는 sns어체로 코멘트를 작성해주세요."},
                    # {"role": "user", "content": "제목:강아지 오늘 공원을 산책하는 도중에 귀여운 강아지를 보았다. 그 강아지는 지금까지 본 강아지 중 가장 귀여운 강아지였다. 그 강아지를 몇 번 쓰다듬었고, 강아지의 초롱초롱한 눈빛이 아주 인상깊었다. 강아지 덕분에 스트레스도 풀고 힐링할 수 있었다. 다음에 기회가 된다면 꼭 강아지를 키우고 싶다."}
                    {"role": "user", "content": text}
                ]
            )
        elif chatbot == 1:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": "당신은 반말을 사용하는 인공지능 챗봇입니다. 입력된 일기를 읽고, 친구에게 써주듯이 코멘트를 100자 이내로 작성해주세요."},
                    # {"role": "system", "content": "당신은 친절한 인공지능 챗봇입니다. 입력된 일기를 읽고,상담사가 상담자에게 상담해주듯이 코멘트를 친절하게 작성해주세요."},
                    # {"role": "system", "content": "당신은 sns어체를 사용하는 인공지능 챗봇입니다. 입력된 일기를 읽고, 실제 사람들이 쓰는 sns어체로 코멘트를 작성해주세요."},
                    # {"role": "user", "content": "제목:강아지 오늘 공원을 산책하는 도중에 귀여운 강아지를 보았다. 그 강아지는 지금까지 본 강아지 중 가장 귀여운 강아지였다. 그 강아지를 몇 번 쓰다듬었고, 강아지의 초롱초롱한 눈빛이 아주 인상깊었다. 강아지 덕분에 스트레스도 풀고 힐링할 수 있었다. 다음에 기회가 된다면 꼭 강아지를 키우고 싶다."}
                    {"role": "user", "content": text}
                ]
            )
        elif chatbot == 2:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    # {"role": "system", "content": "당신은 반말을 사용하는 인공지능 챗봇입니다. 입력된 일기를 읽고, 친구에게 써주듯이 코멘트를 작성해주세요."},
                    # {"role": "system", "content": "당신은 친절한 인공지능 챗봇입니다. 입력된 일기를 읽고,상담사가 상담자에게 상담해주듯이 코멘트를 친절하게 작성해주세요."},
                    {"role": "system", "content": "당신은 sns어체를 사용하는 인공지능 챗봇입니다. 사람들이 sns에서 쓰는 어체로 100자 이내로 코멘트 해주세요"},
                    # {"role": "user", "content": "제목:강아지 오늘 공원을 산책하는 도중에 귀여운 강아지를 보았다. 그 강아지는 지금까지 본 강아지 중 가장 귀여운 강아지였다. 그 강아지를 몇 번 쓰다듬었고, 강아지의 초롱초롱한 눈빛이 아주 인상깊었다. 강아지 덕분에 스트레스도 풀고 힐링할 수 있었다. 다음에 기회가 된다면 꼭 강아지를 키우고 싶다."}
                    {"role": "user", "content": text}
                ]
            )
        elif chatbot ==3:
            completion = client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0613:personal::8YQtjK9S",
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    # {"role": "system", "content": "당신은 반말을 사용하는 인공지능 챗봇입니다. 입력된 일기를 읽고, 친구에게 써주듯이 코멘트를 작성해주세요."},
                    # {"role": "system", "content": "당신은 친절한 인공지능 챗봇입니다. 입력된 일기를 읽고,상담사가 상담자에게 상담해주듯이 코멘트를 친절하게 작성해주세요."},
                    {"role": "system", "content": "당신은 sns어체를 사용하는 인공지능 챗봇입니다."},
                    # {"role": "user", "content": "제목:강아지 오늘 공원을 산책하는 도중에 귀여운 강아지를 보았다. 그 강아지는 지금까지 본 강아지 중 가장 귀여운 강아지였다. 그 강아지를 몇 번 쓰다듬었고, 강아지의 초롱초롱한 눈빛이 아주 인상깊었다. 강아지 덕분에 스트레스도 풀고 힐링할 수 있었다. 다음에 기회가 된다면 꼭 강아지를 키우고 싶다."}
                    {"role": "user", "content": text}
                ]
            )
        comment = completion.choices[0].message.content
        comment = re.sub(r'[.|?|!]', r'.\n', comment)
        self.comment.setText(comment)

    def combobox_slot(self):
        chatbot = self.comboBox.currentIndex()
        return chatbot

    def set_title(self):                 # title 제목 지정
        self.title = self.path[0].split('/')[-1] + " - 힘내 일기"     # 경로에서 파일명만 지정
        self.setWindowTitle(self.title)
        self.edited_flag = False
        self.statusbar.showMessage(self.path[0])

    def text_changed_slot(self):    
        self.edited_flag = True
        self.setWindowTitle('*'+self.title)


    def action_new_slot(self):
        if self.edited_flag:
            ans = QMessageBox.question(self, '저장하기', '저장할까요?',
                                       QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes,
                                       QMessageBox.Yes) # question 박스라 파란 물음표 표식, 따로 씀, QMessageBox.yes는 포커스 지정
            if ans == QMessageBox.Yes:
                if self.action_save_slot():
                    return
            elif ans == QMessageBox.Cancel: return
        self.plainTextEdit.setPlainText('')
        self.path = ['제목 없음', '']
        self.set_title()

    def action_open_slot(self):
        if self.edited_flag:
            ans = QMessageBox.question(self, '저장하기', '저장할까요?',
                                       QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes,
                                       QMessageBox.Yes)
            if ans == QMessageBox.Yes: self.action_save_slot()
            elif ans == QMessageBox.Cancel: return
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(
            self, 'Open file', '', 'Text File(*.txt);; Python File(*.py);; All File(*.*)'
        )
        if self.path[0] != '':
            with open(self.path[0], 'r') as f:
                str_read = f.read()
                self.plainTextEdit.setPlainText(str_read)       # plainText는 setPlainText()로 받아야 함
                self.set_title()
        else: self.path = old_path

    def action_save_slot(self):
        if self.path[0] != '제목 없음':
            with open(self.path[0], 'w') as f:
                f.write(self.plainTextEdit.toPlainText())
            self.set_title()
        else: return self.action_save_as_slot();

    def action_save_as_slot(self):
        old_path = self.path
        self.path = QFileDialog.getSaveFileName(
            self, 'Save file', '', 'Text Files(*.txt);; Python Files(*.py);; All Files(*.*)'
        )       # 경로만 리턴해 줌, 실제로 저장안 함
        print(self.path)
        if self.path[0] != '':
            with open(self.path[0], 'w') as f:
                f.write(self.plainTextEdit.toPlainText())   # plainText는 toplainText()로 읽어야 함
            self.set_title()
            return 0
        else:
            self.path = old_path
            return 1

    def action_exit_slot(self):
        if self.edited_flag:
            ans = QMessageBox.question(self, '저장하기', '저장할까요?',
                                       QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes,
                                       QMessageBox.Yes)
            if ans == QMessageBox.Yes:
                if self.action_save_slot():
                    return
            elif ans == QMessageBox.Cancel:
                return
        self.close()


    def action_font_slot(self):
        font = QFontDialog.getFont()    # ex) (<PyQt5.QtGui.QFont object at 0x00000187B1629510>, True)
        if font[1]:         # ok 를 누르면 True, Cancle 을 누르면 False
            self.plainTextEdit.setFont(font[0]) # Font에 대한 정보, ex) <PyQt5.QtGui.QFont object at 0x00000187B1629510>


    def action_about_slot(self):
        QMessageBox.about(
            self, 'PyQT Notepad', '''만든이 : ABC lab\n\r버전 정보 : 1.0.0''')




if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = OpenAI(
        api_key="api_key input")
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())



