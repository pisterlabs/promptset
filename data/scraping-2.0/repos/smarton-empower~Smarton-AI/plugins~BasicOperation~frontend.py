# Smarton AI for Kicad - Analying help documentation paragraphs and invoke plugins intelligently
# Copyright (C) 2023 Beijing Smarton Empower
# Contact: yidong.tian@smartonep.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import wx
import wx.html2
import openai
import threading
import queue

from .preinput import plugin_preinput, basicop_helpers
from .GPTModels import GPTModel

openai.api_key = ""


class MyBrowser(wx.Dialog):
    def __init__(self, *args, **kwds):
        super(MyBrowser, self).__init__(*args, **kwds)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.browser = wx.html2.WebView.New(self)
        sizer.Add(self.browser, 1, wx.EXPAND, 10)

        self.SetSizer(sizer)
        self.SetSize((800, 600))


class OneCommandLine(wx.Frame):
    def __init__(self, parent, board):
        super(OneCommandLine, self).__init__(parent, title="OneCommandLine", size=(600, 300))

        self.board = board

        screen_width, screen_height = wx.DisplaySize()
        position_x = screen_width // 50 + 600
        position_y = (screen_height - self.GetSize()[1]) // 2
        self.SetPosition((position_x, position_y))

        # 创建聊天窗口的控件
        self.panel = wx.Panel(self)
        # 聊天记录框
        self.chat_display = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        # 用户输入框
        self.text_ctrl = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE)
        # 按钮
        self.send_button = wx.Button(self.panel, label="Send")
        self.clear_button = wx.Button(self.panel, label="Clear")
        # 新增：显示/隐藏插件记录框的按钮
        self.toggle_plugin_display_button = wx.Button(self.panel, label="View all plugins")
        self.show_external_plugin_button = wx.Button(self.panel, label="Use")
        self.show_external_plugin_button.Hide()

        # 插件名称记录框
        self.plugin_display = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        self.plugin_display.Hide()  # 默认隐藏插件记录框

        # 布局控件
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.send_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        button_sizer.Add(self.clear_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        button_sizer.Add(self.toggle_plugin_display_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        button_sizer.Add(self.show_external_plugin_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)

        # 垂直布局
        chat_sizer = wx.BoxSizer(wx.VERTICAL)
        chat_sizer.Add(self.chat_display, proportion=1, flag=wx.EXPAND)
        chat_sizer.Add(self.text_ctrl, proportion=0, flag=wx.EXPAND | wx.ALL)  # 添加边距
        chat_sizer.Add(button_sizer, proportion=0, flag=wx.EXPAND)  # 添加按钮布局

        # 水平布局
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(chat_sizer, proportion=2, flag=wx.EXPAND | wx.RIGHT)  # 添加边距
        sizer.Add(self.plugin_display, proportion=1, flag=wx.EXPAND)

        self.panel.SetSizer(sizer)

        # 设置输入框的最小尺寸
        chat_sizer.SetItemMinSize(self.text_ctrl, (self.text_ctrl.GetMinSize()[0], 50))

        # gpt属性
        self.language = ""
        self.state = "choose_language"
        self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
        self.chat_display.AppendText(f"Smarton AI: Please choose language / 请选择语言: \n")
        self.chat_display.AppendText(f"Smarton AI: choose from (en/zh) / 从(en/zh)中选择: \n")

        self.plugin_names, self.plugin_args, self.plugin_description_msg = plugin_preinput.plugin_preinput()

        self.plugin_display.AppendText("=" * 30 + "\n")
        self.plugin_display.AppendText(f"<< Plugin names/插件名 >>\n")
        self.plugin_display.AppendText("="*30+"\n")
        self.plugin_display.AppendText("-" * 30 + "\n")

        # self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
        # self.chat_display.AppendText(f"Smarton AI: Board: {self.board}\n")
        self.plugin_ids = []
        for i in range(len(self.plugin_names)):
            plugin = self.plugin_names[i]
            args = self.plugin_args[i]
            self.plugin_ids.append(i)
            # self.chat_display.AppendText(f"Plugin {i+1}: {plugin}\n")
            self.plugin_display.AppendText(f"Plugin {i+1}: {plugin}\n")
            self.plugin_display.AppendText("-"*30+"\n")
        # self.chat_display.AppendText(f"\nSmarton AI: We currently have the above plugins, please choose one plugin to use at a time\n")

        # 绑定事件
        self.send_button.Bind(wx.EVT_BUTTON, self.on_submit)
        self.clear_button.Bind(wx.EVT_BUTTON, self.on_clear_button_clicked)
        self.toggle_plugin_display_button.Bind(wx.EVT_BUTTON, self.view_all_plugins)
        self.show_external_plugin_button.Bind(wx.EVT_BUTTON, self.use_external_plugins)

        self.model = 'gpt-3.5-turbo-0613'

        self.userInputQueue = queue.Queue()
        self.exitEvent = threading.Event()

        self.browser_dialog = None

    def on_clear_button_clicked(self, event):
        self.chat_display.SetValue("")  # 清空聊天信息

    def view_all_plugins(self, event):
        # 切换插件记录框的显示状态
        self.plugin_display.Show(not self.plugin_display.IsShown())
        # 调整布局
        self.panel.Layout()

    def use_external_plugins(self, event):
        # run_with_dialog()
        return

    def choose_language(self, user_input):

        language = user_input
        if language == 'zh':
            self.language = language
            self.state = "use_plugin"
            self.welcome()
        elif language == 'en':
            self.language = language
            self.state = "use_plugin"
            self.welcome()
        else:
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(
                f"Smarton AI: not a valid language, please choose from (en/zh) / 不是有效的语言, 请重新选择(en/zh): \n")

        self.text_ctrl.SetValue("")  # 清空输入框
        self.text_ctrl.SetMinSize((self.text_ctrl.GetMinSize()[0], -1))  # 恢复文本框的最小尺寸
        self.Layout()  # 重新布局窗口

    def welcome(self):
        if self.language == 'zh':
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(f"请输入您想用的插件名称, 或者输入想要实现的效果，我们将为您推荐合适的插件\n")

        elif self.language == 'en':
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(f"Please enter the name of the plugin you want to use, or enter the effect you want to achieve, we will recommend the appropriate plugin for you\n")

    def on_submit(self, event):
        try:
            user_input = self.text_ctrl.GetValue()
            self.chat_display.AppendText(f"==========  User  ==========\n")
            self.chat_display.AppendText(f"User: {user_input}\n")

            if self.state == "choose_language":
                self.choose_language(user_input)

            elif self.state == "use_plugin":
                self.text_ctrl.Clear()
                self.userInputQueue.put(user_input)

            self.text_ctrl.SetValue("")  # 清空输入框
            self.text_ctrl.SetMinSize((self.text_ctrl.GetMinSize()[0], -1))  # 恢复文本框的最小尺寸
            self.Layout()  # 重新布局窗口

        except Exception as e:
            self.chat_display.AppendText(f"An exception occurred: {e}\n")

    def argument_msg(self, args, need_click, example_msg):
        if need_click:
            if self.language == 'zh':
                wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                wx.CallAfter(self.chat_display.AppendText, f'=> 请先用鼠标点击选择元件\n')
                if args != '':
                    wx.CallAfter(self.chat_display.AppendText, f'=> 请给出相关参数 {args} 并用逗号分开\n')
                    wx.CallAfter(self.chat_display.AppendText, f'例如：{example_msg}\n')
                wx.CallAfter(self.chat_display.AppendText, f'=> 在发送参数后，请重新点击PCB编辑器对应元件，元件将在再次点击后发生变化\n')
            if self.language == 'en':
                wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                wx.CallAfter(self.chat_display.AppendText, f'=> Please click the footprints first\n')
                wx.CallAfter(self.chat_display.AppendText, f'=> Please provide some arguments {args} and separated by commas\n')
                if args != '':
                    wx.CallAfter(self.chat_display.AppendText, f'eg. {example_msg}\n')
                wx.CallAfter(self.chat_display.AppendText, f'=> After sending the parameters, please click the corresponding footprint in the PCB editor again, the component will change after clicking again\n')
        else:
            if self.language == 'zh':
                wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                if args != '':
                    wx.CallAfter(self.chat_display.AppendText, f'=> 请给出相关参数 {args} 并用逗号分开\n')
                    wx.CallAfter(self.chat_display.AppendText, f'例如：{example_msg}\n')
                wx.CallAfter(self.chat_display.AppendText, f'=> 在发送参数后，请重新点击PCB编辑器对应元件，元件将在再次点击后发生变化\n')
            if self.language == 'en':
                wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                wx.CallAfter(self.chat_display.AppendText,
                             f'=> Please provide some arguments {args} and separated by commas\n')
                if args != '':
                    wx.CallAfter(self.chat_display.AppendText, f'eg. {example_msg}\n')
                wx.CallAfter(self.chat_display.AppendText, f'=> After sending the parameters, please click the corresponding footprint in the PCB editor again, the component will change after clicking again\n')

    def not_valid_argument_msg(self):
        if self.language == 'zh':
            wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
            wx.CallAfter(self.chat_display.AppendText, f'参数输入有误，请重试\n')
        if self.language == 'en':
            wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
            wx.CallAfter(self.chat_display.AppendText, f'not valid arguments, please try again\n')

    def done_msg(self):
        if self.language == 'zh':
            wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
            wx.CallAfter(self.chat_display.AppendText, f'完成，您可以继续选择插件\n')
        if self.language == 'en':
            wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
            wx.CallAfter(self.chat_display.AppendText, f'Done, you can continue to choose a plugin\n')

    def display_url(self, url):
        if self.browser_dialog is not None and self.browser_dialog.IsShown():
            self.browser_dialog.browser.LoadURL(url)
        else:
            self.browser_dialog = MyBrowser(None, -1, "Official Document")
            self.browser_dialog.browser.LoadURL(url)
            self.browser_dialog.Show()

    def handle_user_input(self):
        need_recommend = True
        while True:
            pname = None
            try:
                user_input = self.userInputQueue.get()
                if user_input in self.plugin_names:
                    need_recommend = False
                    pname = user_input
                if need_recommend:
                    messages = self.plugin_description_msg
                    messages.append({"role": "user", "content": f"{user_input}\n"})
                    gpt = GPTModel(self.model, messages, self.plugin_names, self.plugin_ids)
                    self.button_Disable()
                    response = gpt.ask_gpt("Plugin", self.chat_display.AppendText)
                    wx.CallAfter(self.button_Enable, response)

                    selected_plugin_names = response['Plugin']
                    if self.language == 'zh':
                        wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                        wx.CallAfter(self.chat_display.AppendText, f'建议使用: {selected_plugin_names}\n')
                    if self.language == 'en':
                        wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                        wx.CallAfter(self.chat_display.AppendText, f'Recommend Plugin: {selected_plugin_names}\n')
                    pname = self.userInputQueue.get()

                not_valid_plugin = True
                while not_valid_plugin:
                    # plugin 1: rotate_fp_by_fp_name
                    if pname == self.plugin_names[0]:
                        while True:
                            try:
                                self.argument_msg(self.plugin_args[0], False, 'P1, 45')
                                params = self.userInputQueue.get().split(',')
                                footprint_name = params[0].strip()
                                rotate_angle = int(params[1].strip())

                                wx.CallAfter(basicop_helpers.rotate_fp_by_fp_name,
                                    self.board,
                                    footprint_name,
                                    rotate_angle,
                                )
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 2: rotate_fp_by_mouse
                    elif pname == self.plugin_names[1]:
                        while True:
                            try:
                                self.argument_msg(self.plugin_args[1], True, '45')
                                params = self.userInputQueue.get().split(',')
                                rotate_angle = int(params[0].strip())

                                wx.CallAfter(basicop_helpers.rotate_fp_by_mouse,
                                             self.board,
                                             rotate_angle,
                                             )
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 3: move_fp_by_fp_name
                    elif pname == self.plugin_names[2]:
                        while True:
                            try:
                                self.argument_msg(self.plugin_args[2], False, 'P1, 1000000, 1000000')
                                params = self.userInputQueue.get().split(',')
                                footprint_name = params[0].strip()
                                X_offset = int(params[1].strip())
                                Y_offset = int(params[2].strip())

                                wx.CallAfter(basicop_helpers.move_fp_by_fp_name,
                                             self.board,
                                             footprint_name,
                                             X_offset,
                                             Y_offset,
                                             )
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 4: move_fp_by_mouse
                    elif pname == self.plugin_names[3]:
                        while True:
                            try:
                                self.argument_msg(self.plugin_args[3], True, '1000000, 1000000')
                                params = self.userInputQueue.get().split(',')
                                X_offset = int(params[0].strip())
                                Y_offset = int(params[1].strip())

                                wx.CallAfter(basicop_helpers.move_fp_by_mouse,
                                             self.board,
                                             X_offset,
                                             Y_offset,
                                             )
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 5: flip_fp_by_fp_name
                    elif pname == self.plugin_names[4]:
                        while True:
                            try:
                                self.argument_msg(self.plugin_args[4], False, 'P1')
                                params = self.userInputQueue.get().split(',')
                                footprint_name = params[0].strip()

                                wx.CallAfter(basicop_helpers.flip_fp_by_fp_name,
                                             self.board,
                                             footprint_name,
                                             )
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 6: flip_fp_by_mouse
                    elif pname == self.plugin_names[5]:
                        while True:
                            try:
                                self.argument_msg(self.plugin_args[5], True, '')

                                wx.CallAfter(basicop_helpers.flip_fp_by_mouse,
                                             self.board,
                                             )
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 7: board_to_pdf
                    elif pname == self.plugin_names[6]:
                        while True:
                            try:
                                wx.CallAfter(basicop_helpers.board_to_pdf)
                                # wx.CallAfter(self.display_url, "https://gitlab.com/dennevi/Board2Pdf/-/blob/main/README.md")
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 8: place_footprints
                    elif pname == self.plugin_names[7]:
                        while True:
                            try:
                                wx.CallAfter(basicop_helpers.place_footprints)
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 9: replicate_layout
                    elif pname == self.plugin_names[8]:
                        while True:
                            try:
                                wx.CallAfter(basicop_helpers.replicate_layout)
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 10: save_restore_layout
                    elif pname == self.plugin_names[9]:
                        while True:
                            try:
                                wx.CallAfter(basicop_helpers.save_restore_layout)
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # plugin 11: schematic_positions_to_layout
                    elif pname == self.plugin_names[10]:
                        while True:
                            try:
                                wx.CallAfter(basicop_helpers.schematic_positions_to_layout)
                                break
                            except:
                                self.not_valid_argument_msg()

                        self.done_msg()
                        not_valid_plugin = False
                        need_recommend = True

                    # invalid plugin name
                    else:
                        if self.language == 'zh':
                            wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                            wx.CallAfter(self.chat_display.AppendText, f'不是一个有效的插件名，请从下面的插件中选择\n{self.plugin_names}\n')

                        if self.language == 'en':
                            wx.CallAfter(self.chat_display.AppendText, f'==========  Smarton AI  ==========\n')
                            wx.CallAfter(self.chat_display.AppendText, f'Not a valid plugin name, please give a valid name from the following\n{self.plugin_names}\n')
                        pname = self.userInputQueue.get()

            except Exception as e:
                self.chat_display.AppendText(f"An exception occurred: {e}\n")



    def button_Enable(self, response):
        text = self.chat_display.GetValue()
        new_text = ''
        if self.language == 'zh':
            new_text = text.replace('正在匹配相应的插件，请稍候 ...', '')
        if self.language == 'en':
            new_text = text.replace('Matching the appropriate Plugin, please wait for a minute ...', '')
        self.chat_display.SetValue(new_text)

        if self.language == 'zh':
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(f"根据您的要求，我们向您推荐这些插件\n")
        if self.language == 'en':
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(f"Based on your request, we recommend you these plugins\n")
        self.chat_display.AppendText(f"{response}\n")
        self.send_button.Enable()

    def button_Disable(self):
        if self.language == 'zh':
            wx.CallAfter(self.chat_display.AppendText, f'正在匹配相应的插件，请稍候 ...')
        if self.language == 'en':
            wx.CallAfter(self.chat_display.AppendText, f'Matching the appropriate Plugin, please wait for a minute ...')
        self.send_button.Disable()

