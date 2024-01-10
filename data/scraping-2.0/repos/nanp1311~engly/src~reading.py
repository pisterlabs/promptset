#!/usr/bin/python3
import wx
import openai
import platform
from libs.common import json_open, json_write, path, set_font, add_word

class MyScrollableWindow(wx.ScrolledWindow):
    def __init__(self, parent):
        wx.ScrolledWindow.__init__(self, parent, style=wx.VSCROLL)
        self.SetVirtualSize((2000, 2000))
        self.SetScrollbars(20, 20, 1, 1)

        # テキストボックスの内容を入れる変数
        self.input_words = ""
        self.input_trans = ""
        # ファイルパスの指定
        self.file_words = path("words.json")
        self.file_response = path("response.json")
        # カウントのための変数
        self.count = 0
        # 必要なやつ
        #self.Bind(wx.EVT_CLOSE, self.onExit)
        self.__create_widget()
        self.__do_layout()

    # Widgetを作成するメソッド
    def __create_widget(self):
        self.SetBackgroundColour((224, 224, 224))
        # テキストボックス
        self.txtCtrl = wx.TextCtrl(self, -1, style=wx.TE_MULTILINE, size=(500, 80))
        self.txtCtrl.SetDefaultStyle(wx.TextAttr(wx.BLACK))
        self.txtCtrl.SetFont(set_font(15))
        
        # wordボタン
        self.btn_word = wx.Button(self, label="word")
        self.btn_word.SetForegroundColour('#000000')
        self.btn_word.Bind(wx.EVT_BUTTON, self.push_word)

        # transボタン
        self.btn_trans = wx.Button(self, label="trans")
        self.btn_trans.SetForegroundColour('#000000')
        self.btn_trans.Bind(wx.EVT_BUTTON, self.push_trans)

        # 返答文表示
        self.txt = wx.StaticText(self, -1, "", style=wx.TE_LEFT)
        self.txt.SetMinSize((-1, 300))
        self.txt.SetForegroundColour('#000000')
        self.txt.SetFont(set_font(15))

    # レイアウトを設定するメソッド
    def __do_layout(self):
        # 各sizer定義
        self.sizer_all = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_btn = wx.BoxSizer(wx.VERTICAL)
        self.sizer_txt = wx.BoxSizer(wx.VERTICAL)
        self.sizer_word = wx.BoxSizer(wx.VERTICAL)
        
        # テキストボックス
        self.sizer.Add(self.txtCtrl, flag=wx.ALIGN_CENTER | wx.ALL, border=5)
        # word, transボタン
        self.sizer_btn.Add(self.btn_word, flag=wx.ALIGN_CENTER | wx.TOP, border=5)
        self.sizer_btn.Add(self.btn_trans, flag=wx.ALIGN_CENTER | wx.TOP, border=5)
        # テキストボックスとボタンを結合
        self.sizer.Add(self.sizer_btn, flag=wx.ALIGN_CENTER | wx.ALL, border=0)
        self.sizer_all.Add(self.sizer, flag=wx.ALIGN_LEFT | wx.ALL, border=30)

        # 単語保存用ボタン、返答表示用テキストを結合
        self.sizer_txt.Add(self.sizer_word, flag=wx.ALIGN_LEFT | wx.TOP, border=0)
        self.sizer_txt.Add(self.txt, flag=wx.ALIGN_LEFT | wx.TOP, border=20)

        # 全てを合体しセット
        self.sizer_all.Add(self.sizer_txt, flag=wx.ALIGN_LEFT | wx.LEFT, border=50)
        self.SetSizer(self.sizer_all)

    # ボタン押したときの処理
    def push_word(self, event):
        flag = False
        # テキストボックスが更新されていれば返答をリセット
        if self.txtCtrl.GetValue().strip() != self.input_words:
            self.input_words = self.txtCtrl.GetValue().strip()
            flag = True
        # テキストボックスの内容を受け取りChatGPTに質問
        if flag:
            sys = "あなたは英語が得意なアシスタントです。"
            assi = "英単語の意味を日本語で簡潔に回答します。"
            usr = self.input_words + "この英文に出てくる単語や熟語の意味をword: meaningという形で日本語で表示してください。"
            # 単語の場合は品詞も共に日本語で表示してください。
            response = self.api_response(sys, assi, usr)
            if not response:
                self.txt.SetLabel("APIキーが間違っています。")
            else:
                # ChatGPTの返答をjsonファイルに保存
                self.write_response("Word", self.input_words, response)
                # 返答に含まれる単語と意味をボタン化
                # 古いボタンを削除
                for i in range(0, self.count):
                    exec("self.btn_{}.Destroy()".format(i))
                all_word = response.replace("- ", "").split('\n') #前に書かれてる方から処理する
                self.count = 0
                for word in all_word:
                    if ": " in word:
                        exec("self.btn_{} = wx.Button(self, label=word)".format(self.count))
                        exec("self.btn_{}.Bind(wx.EVT_BUTTON, self.push_add)".format(self.count))
                        exec("self.btn_{}.SetForegroundColour('#000000')".format(self.count))
                        exec("self.sizer_word.Add(self.btn_{}, flag=wx.ALIGN_LEFT | wx.TOP, border=10)".format(self.count))
                        self.count += 1
        self.FitInside()
        self.Layout()
    
    def push_add(self, event):
        # 押したボタンの情報を取得
        btn = event.GetEventObject()
        # ボタンのラベルを取得し分解
        content = btn.GetLabel().split(': ')
        # 単語と意味の保存
        add_word(content[0].strip(), content[1].strip(), content[0].strip(), self.file_words)
        # ボタンを押せるのは一度だけ
        btn.Disable()

    def push_trans(self, event):
        flag = False
        # テキストボックスが更新されていれば返答をリセット
        if self.txtCtrl.GetValue().strip() != self.input_trans:
            self.input_trans = self.txtCtrl.GetValue().strip()
            flag = True
        # テキストボックスの内容を受け取りChatGPTに質問
        if flag:
            sys = "あなたは英語が得意なアシスタントです。"
            assi = "英文を丁寧に翻訳します。"
            usr = self.input_trans + "この英文の日本語訳を表示してください。それに加え、この英文の主語と動詞を英語で表示してください。" #さらに、この英文の文法について解説してください。"
            response = self.api_response(sys, assi, usr)
            if not response:
                self.txt.SetLabel("APIキーが間違っています。")
            else:
                # ChatGPTの返答をjsonファイルに保存
                self.write_response("Translation", self.input_trans, response)
                # ChatGPTの返答を表示
                self.txt.SetLabel(response)
        self.FitInside()
        self.Layout()

    # ChatGPTに質問
    def api_response(self, system="", assistant="", user=""):
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": system},
                    {"role": "assistant", "content": assistant},
                    {"role": "user", "content": user}
            ],
            temperature=0,
            top_p = 0
            )
            return response['choices'][0]['message']['content'].strip()
        except:
            return False

    # ChatGPTの返答をjsonファイルに保存
    def write_response(self, tag, sentence, response):
        json_data = json_open(self.file_response)
        new_data = {
                    "tag": tag,
                    "sentence": sentence,
                    "response": response
                }
        json_data["ChatGPT"].append(new_data)
        json_write(self.file_response, json_data)

class SampleFrame(wx.Frame):
    def __init__(self, parent, ID, title):
        self.system = platform.system()
        if self.system == "Windows":
            self.x = 720
            self.y = 480
            self.enc = 'cp932'
        else:
            self.x = 800
            self.y = 600
            self.enc = 'utf-8'
        wx.Frame.__init__(self, parent, title=title, pos=(0, 0), size=(self.x, self.y))
        scrollable_window = MyScrollableWindow(self)
        self.__set_apikey()

    def __set_apikey(self):
        with open(path("apikey", "api"), "r", encoding=self.enc, errors='ignore') as f:
            openai.api_key = f.read().strip()

    # xボタン押下時の処理
    def onExit(self, event):
        dlg = wx.MessageDialog(self, "プログラムを終了しますか？", "確認", wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            self.Destroy()  # ウィンドウを破棄してプログラムを終了
        else:
            dlg.Destroy()

# アプリケーションクラス
class SampleApp(wx.App):
    def OnInit(self):
        frame = SampleFrame(None, -1, "Reading")
        self.SetTopWindow(frame)
        frame.Centre()
        frame.Show(True)
        return True

# メイン
#if __name__ == '__main__':
def main():
    app = SampleApp()
    app.MainLoop()