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
import openai
import wx.html2
from bs4 import BeautifulSoup
import os
import webbrowser
import wx.html2 as webview
import json
import threading
import queue
import atexit

from .GPTModels import MainGPT, SubGPT, GPTModel, GPTModel_QA
from .preinput import main_gpt_preinput, sub_gpt_preinput

openai.api_key = ""


class WebViewWindow(wx.Frame):
    def __init__(self, parent):
        super(WebViewWindow, self).__init__(parent, title="Web View", size=(800, 600))
        self.web_view = webview.WebView.New(self)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.web_view, proportion=1, flag=wx.EXPAND)  # Add WebView to the new window
        self.SetSizer(sizer)


class ChatWindow(wx.Frame):
    def __init__(self, parent):
        super(ChatWindow, self).__init__(parent, title="Chat Window", size=(700, 700))

        screen_width, screen_height = wx.DisplaySize()
        position_x = screen_width // 50
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
        self.display_button = wx.Button(self.panel, label="View Result")
        # self.web_view = webview.WebView.New(self.panel)

        # 布局控件
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.send_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        button_sizer.Add(self.clear_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        button_sizer.Add(self.display_button, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.chat_display, proportion=1, flag=wx.EXPAND)
        sizer.Add(self.text_ctrl, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)  # 添加边距
        sizer.Add(button_sizer, proportion=0, flag=wx.EXPAND)  # 添加按钮布局
        # sizer.Add(self.web_view, proportion=1, flag=wx.EXPAND)  # 添加WebView
        self.panel.SetSizer(sizer)

        # 设置输入框的最小尺寸
        sizer.SetItemMinSize(self.text_ctrl, (self.text_ctrl.GetMinSize()[0], 50))

        # gpt属性
        self.language = ""
        self.path_active_eeschema = ""
        self.path_active_pcbnew = ""

        self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
        self.chat_display.AppendText(f"Smarton AI: Please choose language / 请选择语言: \n")
        self.chat_display.AppendText(f"Smarton AI: choose from (en/zh) / 从(en/zh)中选择: \n")

        # 绑定事件
        self.send_button.Bind(wx.EVT_BUTTON, self.on_submit)
        self.clear_button.Bind(wx.EVT_BUTTON, self.on_clear_button_clicked)
        self.display_button.Bind(wx.EVT_BUTTON, self.display_result)

        self.state = "choose_language"
        self.model = 'gpt-3.5-turbo-0613'

        self.userInputQueue = queue.Queue()
        self.exitEvent = threading.Event()

        self.result_dic = None

    def on_clear_button_clicked(self, event):
        self.chat_display.SetValue("")  # 清空聊天信息

    def display_result(self, event):
        try:
            # self.result_dic = {
            #     '12-Introduction to the KiCad PCB Editor': [
            #         {'1-Initial configuration': self.path_active_pcbnew + '01_1.html'},
            #         {'3-Navigating the editing canvas': self.path_active_pcbnew + '01_3.html'}
            #     ],
            #     '13-Display and selection controls': [
            #         {'4-Net highlighting': self.path_active_pcbnew + '02_4.html'}
            #     ]
            # }

            if self.result_dic is None or self.result_dic == {}:
                self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
                self.chat_display.AppendText(f"Result dic is empty\n")
            elif self.language == "":
                self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
                self.chat_display.AppendText(f"Please choose language first (en/zh) / 请选择语言 (en/zh):\n")
            else:
                file_path_dic = list(self.result_dic.values())
                html_file_paths = []
                for dic_list in file_path_dic:
                    for dic in dic_list:
                        html_file_paths += list(dic.values())

                self.browser_display(html_file_paths, self.path_active_pcbnew)

        except Exception as e:
            self.chat_display.AppendText(f"An exception occurred: {e}\n")

    def browser_display(self, html_file_paths, rel_path):
        self.web_view_window = WebViewWindow(self)
        self.web_view = self.web_view_window.web_view
        # 清空WebView内容
        self.web_view.ClearBackground()

        # Read the first HTML file and parse it with BeautifulSoup
        with open(html_file_paths[0], 'r') as file:
            soup_first = BeautifulSoup(file, 'html.parser')

        # Find the sectionbody in the first HTML file
        sectionbody_first = soup_first.find('div', {'id': 'preamble'})

        # Iterate over the rest of the HTML files and add their sectionbody to the first one
        for file_path in html_file_paths[1:]:
            with open(file_path, 'r') as file:
                soup_other = BeautifulSoup(file, 'html.parser')

            sectionbody_other = soup_other.find('div', {'class': 'sectionbody'})

            # Add the sectionbody of other HTML files to the first one
            sectionbody_first.append(sectionbody_other)

        # Add jquery and bootstrap cdn to head
        head = soup_first.head
        jquery_cdn = soup_first.new_tag('script',
                                        src='https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js')
        bootstrap_cdn = soup_first.new_tag('script',
                                           src='https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js')
        head.append(jquery_cdn)
        head.append(bootstrap_cdn)

        # Add pagination script to body
        pagination_script = """
            <script>
                $(document).ready(function () {
                    var total_pages = $('.sectionbody').length;
                    var current_page = 1;
                    $('.sectionbody').hide();
                    $('.sectionbody:nth-child(' + current_page + ')').show();

                    $('#previous').click(function () {
                        if (current_page > 1) {
                            current_page--;
                            $('.sectionbody').hide();
                            $('.sectionbody:nth-child(' + current_page + ')').show();
                            window.scrollTo(0, 0);  // Scroll to top of the page
                        }
                    });

                    $('#next').click(function () {
                        if (current_page < total_pages) {
                            current_page++;
                            $('.sectionbody').hide();
                            $('.sectionbody:nth-child(' + current_page + ')').show();
                            window.scrollTo(0, 0);  // Scroll to top of the page
                        }
                    });
                });
            </script>
            """

        pagination_button = """
        <div id="navigation" style="position:fixed; top:50%; left:0; z-index:100;">
            <button id="previous" class="navigation-button">&#8679;</button>
            <br>
            <button id="next" class="navigation-button">&#8681;</button>
        </div>
        """

        navigation_css = """
        <style>
        body {
            padding-left: 20px;  /* Increase left padding of the body */
        }

        #navigation {
            left: 20px;  /* Move the navigation buttons a bit to the right */
        }

        .navigation-button {
            background-color: #f8f8f8; /* Light gray */
            border: none;
            color: black;
            padding: 10px 15px;  /* Make buttons narrower */
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 15px;
            transition: background-color 0.3s;
            margin-bottom: 20px;  /* Increase vertical distance between buttons */
            margin-left: 10px;
        }

        .navigation-button:hover {
            background-color: #555; /* Gray */
            color: white;
        }
        </style>
        """

        # ...

        # Add the CSS to the head of the HTML
        soup_first.head.append(BeautifulSoup(navigation_css, 'html.parser'))

        # Add the navigation buttons to the body of the HTML
        soup_first.body.insert(0, BeautifulSoup(pagination_button, 'html.parser'))

        body = soup_first.body
        # body.insert(0, BeautifulSoup(pagination_button, 'html.parser'))
        body.append(BeautifulSoup(pagination_script, 'html.parser'))

        # Create the temp HTML file with the combined content
        temp_html_path = rel_path + "temp.html"
        with open(temp_html_path, 'w') as temp_file:
            temp_file.write(str(soup_first))

        # 在WebView中加载临时HTML文件
        temp_html_absolute_path = os.path.abspath(temp_html_path)
        self.web_view.LoadURL("file://" + temp_html_absolute_path)

        # 获取ChatWindow的位置和大小
        chat_window_position = self.GetPosition()
        chat_window_size = self.GetSize()

        # 设置WebViewWindow的位置在ChatWindow的右边
        web_view_window_position = (chat_window_position[0] + 2 * chat_window_size[0] // 3, chat_window_position[1])
        self.web_view_window.SetPosition(web_view_window_position)
        self.web_view_window.Show()

        # 删除临时HTML文件
        # os.remove(temp_html_path)

    def choose_language(self, user_input):

        language = user_input

        if language == 'zh':
            self.path_active_pcbnew = '/Users/alain/Documents/KiCad/7.0/3rdparty/plugins/chat/static/html/zh/pcbnew_zh/'
            self.path_active_eeschema = '/Users/alain/Documents/KiCad/7.0/3rdparty/plugins/chat/static/html/zh/eeschema_zh/'
            self.language = language
            # change event
            # self.send_button.Bind(wx.EVT_BUTTON, self.main_sub_gpt)
            self.state = "ask_main_gpt"
            self.main_sub_msg()
        elif language == 'en':
            self.path_active_pcbnew = '/Users/alain/Documents/KiCad/7.0/3rdparty/plugins/chat/static/html/en/pcbnew/'
            self.path_active_eeschema = '/Users/alain/Documents/KiCad/7.0/3rdparty/plugins/chat/static/html/en/eeschema/'
            self.language = language
            # change event
            # self.send_button.Bind(wx.EVT_BUTTON, self.main_sub_gpt)
            self.state = "ask_main_gpt"
            self.main_sub_msg()
        else:
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(
                f"Smarton AI: not a valid language, please choose from (en/zh) / 不是有效的语言, 请重新选择(en/zh): \n")

        self.create_main_sub_gpt()
        self.text_ctrl.SetValue("")  # 清空输入框
        self.text_ctrl.SetMinSize((self.text_ctrl.GetMinSize()[0], -1))  # 恢复文本框的最小尺寸
        self.Layout()  # 重新布局窗口

    def main_sub_msg(self):
        if self.language == 'zh':
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(f"请输入您关于Kicad的需求或问题\n")

        elif self.language == 'en':
            self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
            self.chat_display.AppendText(f"Please input your request or question about Kicad\n")

    def create_main_sub_gpt(self):

        Task_plugins = sub_gpt_preinput.subtask_html_path(self.path_active_eeschema, self.path_active_pcbnew)
        subtask_id_lists = [[i for i in range(len(Task_plugins[i]))] for i in range(len(Task_plugins))]
        subtasks_name_all, task_subgpt_messages_list = sub_gpt_preinput.subtask_preiput()
        gpts = []

        for subgpt_messages, subtasks_name, subtask_id_list in zip(task_subgpt_messages_list, subtasks_name_all,
                                                                   subtask_id_lists):
            gpts.append(GPTModel('gpt-3.5-turbo-0613', subgpt_messages, language=self.language, topics=subtasks_name,
                                 id_list=subtask_id_list))

        sub_gpts = []
        for sub_gpt, subtasks, subtasks_name in zip(gpts, Task_plugins, subtasks_name_all):
            sub_gpts.append(SubGPT(sub_gpt, subtasks, subtasks_name))

        tasks, main_topics, main_gpt_messages = main_gpt_preinput.main_gpt_preinput()
        main_task_ids = [i for i in range(len(main_topics))]

        main_gpt_model = GPTModel('gpt-3.5-turbo-0613', main_gpt_messages, self.language, main_topics, main_task_ids)
        main_gpt = MainGPT(main_gpt_model, sub_gpts, self.language)

        self.main_gpt = main_gpt

    def on_submit(self, event):
        try:
            user_input = self.text_ctrl.GetValue()
            self.chat_display.AppendText(f"==========  User  ==========\n")
            self.chat_display.AppendText(f"User: {user_input}\n")

            if self.state == "choose_language":
                self.choose_language(user_input)

            elif self.state == "ask_main_gpt":
                self.text_ctrl.Clear()
                self.userInputQueue.put(user_input)

            self.text_ctrl.SetValue("")  # 清空输入框
            self.text_ctrl.SetMinSize((self.text_ctrl.GetMinSize()[0], -1))  # 恢复文本框的最小尺寸
            self.Layout()  # 重新布局窗口

        except Exception as e:
            self.chat_display.AppendText(f"An exception occurred: {e}\n")

    def handle_user_input(self):
        result_dic = {}
        # 主子GPT
        not_terminate = True
        switch = None
        main_task_description = ""
        main_task_description_backup = ""
        while not_terminate:
            try:
                if switch is None:
                    # 询问用户是对主任务的描述
                    main_task_description = self.userInputQueue.get()
                    main_task_description_backup = main_task_description
                if switch == 'r':
                    main_task_description = main_task_description_backup
                self.main_gpt.gpt_model.messages.append({"role": "user", "content": f"{main_task_description}"})
                # 禁用send按钮，并在对话框中显示等待回复，直到GPT返回结果
                self.button_Disable()
                # 将用户的描述发给GPT，并等待回复
                maingpt_response = self.main_gpt.gpt_model.ask_gpt('Task')
                # 激活send按钮，并清除对话框中的等待回复
                wx.CallAfter(self.button_Enable, maingpt_response)
                task_ids, chosen_tasks, chosen_reason = maingpt_response['id'], maingpt_response['Task'], maingpt_response['Reason']
                for i in range(len(task_ids)):
                    task_id = task_ids[i]
                    task_id = int(task_id) - 1
                    chosen_task = chosen_tasks[i]
                    if self.language == 'en':
                        wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                        wx.CallAfter(self.chat_display.AppendText, f"Smarton AI suggests doing task: {chosen_task}. \nReason: {chosen_reason} \nDo you agree? (yes/no)\n")
                    if self.language == 'zh':
                        wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                        wx.CallAfter(self.chat_display.AppendText, f"Smarton AI建议执行任务:{chosen_task}。\n理由:{chosen_reason} \n你同意吗?(yes / no)”)\n")

                    # 询问用户是否选取该主任务
                    agree_task = self.userInputQueue.get()
                    if agree_task != "no":
                        not_terminate = False
                        result_dic[chosen_task] = []
                        wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")

                        if self.language == 'en':
                            wx.CallAfter(self.chat_display.AppendText, f"These are all the subtasks in {chosen_task}:\n")
                        if self.language == 'zh':
                            wx.CallAfter(self.chat_display.AppendText, f"这是所有在{chosen_task}中的子任务:\n")
                        subtask_names = self.main_gpt.subgpts[task_id].subtasks_name

                        for i in range(len(subtask_names)):
                            wx.CallAfter(self.chat_display.AppendText, f"subtask {i + 1}: {subtask_names[i]}\n")

                        # 询问用户对子任务的需求
                        if self.language == 'en':
                            wx.CallAfter(self.chat_display.AppendText, f"If you would like me to recommend some subtasks, please provide more details about task: {chosen_task}, or type j to skip and select the subtask directly\n")
                        if self.language == 'zh':
                            wx.CallAfter(self.chat_display.AppendText, f"如果你想让我推荐一些子任务, 请提供更多关于task: {chosen_task}的细节, 或者输入j跳过, 直接选择子任务\n")

                        subtask_description = self.userInputQueue.get()
                        chosen_subtasks = []

                        if subtask_description == "j":
                            if self.language == 'en':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, "Please enter the number of the subtask you want to know more about, separated by commas (e.g. 1,2,3): \n")
                            if self.language == 'zh':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, "请输入您想进一步了解的子任务的编号, 每个编号之间以逗号隔开(如:1,2,3): \n")

                            # 获取用户选择的子任务
                            selected_subtasks = self.userInputQueue.get()
                            chosen_index, chosen_subtasks = self.generate_checkbox(self.main_gpt.subgpts[task_id].subtasks_name, selected_subtasks)
                            results = self.main_gpt.subgpts[task_id].id_to_html_result(chosen_index)

                            wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                            wx.CallAfter(self.chat_display.AppendText, f"Your choices are: {chosen_subtasks}\n")

                        else:
                            # 禁用send按钮，并在对话框中显示等待回复，直到GPT返回结果
                            self.button_Disable()
                            chosen_subtasks, chosen_subtask_reason, results = self.main_gpt.subgpts[task_id].receive_request(subtask_description)
                            # 激活send按钮，并清除对话框中的等待回复
                            wx.CallAfter(self.button_Enable, chosen_subtasks, chosen_subtask_reason)
                            if self.language == 'en':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, "Please enter the number of the subtask you want to know more about, separated by commas (e.g. 1,2,3): \n")
                            if self.language == 'zh':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, "请输入您想进一步了解的子任务的编号, 每个编号之间以逗号隔开(如:1,2,3): \n")

                            # 获取用户选择的子任务
                            selected_subtasks = self.userInputQueue.get()
                            chosen_index, chosen_subtasks = self.generate_checkbox(self.main_gpt.subgpts[task_id].subtasks_name, selected_subtasks)
                            results = self.main_gpt.subgpts[task_id].id_to_html_result(chosen_index)

                            wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                            wx.CallAfter(self.chat_display.AppendText, f"Your choices are: {chosen_subtasks}\n")

                        # 在result_dic中添加需要的html页面
                        for j in range(len(chosen_subtasks)):
                            chosen_subtask = chosen_subtasks[j]
                            result = results[j]
                            result_dic[chosen_task].append({chosen_subtask: result})

                        # 如果有一个主task, 询问用户是终止并返回结果还是重选主任务
                        if len(task_ids) == 1:

                            if self.language == 'en':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, "Do you want to go back to rechoose tasks(r) or terminate(t)? Please input r/t:\n")
                                switch = self.userInputQueue.get()
                            if self.language == 'zh':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, "您要返回重新选择任务(r)还是终止(t)? 请输入 r/t:\n")
                                switch = self.userInputQueue.get()

                            if switch == 'r':
                                not_terminate = True
                                if self.language == 'en':
                                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"Please provide a reason for not choosing this task: {chosen_task}:\n")
                                if self.language == 'zh':
                                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"请提供不选择此任务的原因:{chosen_task}:\n")
                                rechoose_reason = self.userInputQueue.get()
                                self.main_gpt.gpt_model.messages.append({"role": "system", "content": f"user ask {main_task_description}, user do not agree with choosing this task {chosen_task}, and the reason is: {rechoose_reason}"})
                                break

                            if switch == 't':
                                not_terminate = False
                                # wx.CallAfter(self.chat_display.AppendText, f"==========  result_dic  ==========\n")
                                # wx.CallAfter(self.chat_display.AppendText, f"{result_dic}\n")
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                if self.language == 'en':
                                    wx.CallAfter(self.chat_display.AppendText, f"Thanks for using our service.\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"You can click the <View Result> Button now for detail\n")
                                if self.language == 'zh':
                                    wx.CallAfter(self.chat_display.AppendText, f"感谢您使用我们的服务\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"你可以点击<View Result>按钮来查看详情\n")

                                self.result_dic = result_dic
                                break

                        # 如果有多个主task, 询问用户是终止并返回结果，还是重选主任务，还是继续处理下一个主task
                        else:

                            if self.language == 'en':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, f"Do you want to go back to rechoose tasks(r), go forwarding(g) or terminate(t)? Please input r/g/t:\n")
                                switch = self.userInputQueue.get()
                            if self.language == 'zh':
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                wx.CallAfter(self.chat_display.AppendText, f"您要返回重新选择任务(r)、继续选择(g)还是终止(t)? 请输入 r/g/t:\n")
                                switch = self.userInputQueue.get()

                            if switch == 'r':
                                not_terminate = True
                                if self.language == 'en':
                                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"Please provide a reason for not choosing this task: {chosen_task}:\n")
                                if self.language == 'zh':
                                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"请提供不选择此任务的原因:{chosen_task}:\n")
                                rechoose_reason = self.userInputQueue.get()
                                self.main_gpt.gpt_model.messages.append({"role": "system", "content": f"user ask {main_task_description}, user do not agree with choosing this task {chosen_task}, and the reason is: {rechoose_reason}"})
                                break

                            if switch == 'g':
                                if self.language == 'en':
                                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"go forward to the next main task\n")
                                if self.language == 'zh':
                                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"继续进行下一个主任务\n")

                            if switch == 't':
                                not_terminate = False
                                # wx.CallAfter(self.chat_display.AppendText, f"==========  result_dic  ==========\n")
                                # wx.CallAfter(self.chat_display.AppendText, f"{result_dic}\n")
                                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                                if self.language == 'en':
                                    wx.CallAfter(self.chat_display.AppendText, f"Thanks for using our service.\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"You can click the <View Result> Button now for detail\n")
                                if self.language == 'zh':
                                    wx.CallAfter(self.chat_display.AppendText, f"感谢您使用我们的服务\n")
                                    wx.CallAfter(self.chat_display.AppendText, f"你可以点击<View Result>按钮来查看详情\n")

                                self.result_dic = result_dic
                                break

                    else:
                        self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
                        if self.language == 'en':
                            self.chat_display.AppendText(f"Sorry, can you provide some reason:\n")
                        if self.language == 'zh':
                            self.chat_display.AppendText(f"抱歉，能提供一些原因吗:\n")
                        user_reason_for_task = self.userInputQueue.get()
                        self.main_gpt.gpt_model.messages.append({"role": "system", "content": f"user ask {main_task_description}, user do not agree with choosing this task {chosen_task}, and the reason is: {user_reason_for_task}"})

                        switch = 'r'

                if switch == 'r':
                    # 让GPT重新选择主任务
                    result_dic = {}
                    self.result_dic = result_dic
                    not_terminate = True
                    continue

                wx.CallAfter(self.chat_display.AppendText, f"==========  ALL DONE  ==========\n")
                wx.CallAfter(self.chat_display.AppendText, f"Enter 'qa' to QA_GPT or Enter '' to continue on MAIN_SUB_GPT\n")

                skip_main_sub_task = self.userInputQueue.get()  # 再次等待用户的输入，无限等待
                if skip_main_sub_task == 'qa':
                    # self.chat_display.SetValue("")
                    break
                else:
                    result_dic = {}
                    self.result_dic = result_dic
                    not_terminate = True
                    switch = None
                    if self.language == 'zh':
                        wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                        wx.CallAfter(self.chat_display.AppendText, f"请输入您关于Kicad的需求或问题\n")

                    elif self.language == 'en':
                        wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                        wx.CallAfter(self.chat_display.AppendText, f"Please input your request or question about Kicad\n")

            except Exception as e:
                self.chat_display.AppendText(f"An exception occurred: {e}\n")

        # QA_GPT
        try:
            # process QA_GPT preinput based on the result_dic
            html_paths = []
            for key in list(result_dic.keys()):
                task_list = result_dic[key]
                for task_dic in task_list:
                    for subtask_key in list(task_dic.keys()):
                        html_paths.append(task_dic[subtask_key])

            from bs4 import BeautifulSoup

            html_text_content_list = []
            for html_path in html_paths:
                with open(html_path) as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    div_element = soup.find('div', class_='sect2')
                    html_text_content = div_element.get_text()
                    html_text_content_list.append(html_text_content)

            QA_gpt_messages = [
                {"role": "system",
                 "content": "You are a helpful assistant from Smarton Company, which is designed to help pcb engineers to learn Schematic Editor and PCB Editor for Kicad."},
                {"role": "system",
                 "content": "You should answer the question mainly based on the following content, if the content does not mention the user's question, you can take your own knowledge to the consideration necessarily."},
            ]

            # compute the number of content for each page
            # num_of_content = 3600 // len(html_text_content_list)
            num_of_content = 1000
            QA_gpt_messages += [{"role": "system", "content": f"{html_text_content[:num_of_content]}"} for html_text_content in html_text_content_list]

            QA_res = 'I have understood your request and this is the answer to your previous question"{main_task_description}\n'
            QA_ask = "I am the Smarton AI with understooding your previous requirements, you can ask me further questions, please enter exit to exit\n"
            end_msg = 'Thank you for using our service\n'
            if self.language == "zh":
                QA_res = f'我已经了解您的需求, 这是对您之前的问题"{main_task_description}"的回答\n'
                QA_ask = f"您可以对我进一步提问, 可以在两个问题后输入'add'对我进行信息增量, 退出请输入'exit'\n"
                end_msg = f"感谢您使用我们的服务\n"
            if self.language == "en":
                QA_res = f'I have understood your request and this is the answer to your previous question"{main_task_description}"\n'
                QA_ask = f"You can ask me further questions. You can enter 'add' after two questions to increment my information. To exit, please enter 'exit'\n"
                end_msg = f'Thank you for using our service\n'

            # init QA_GPT
            QA_model = GPTModel_QA(model='gpt-3.5-turbo-0613', messages=QA_gpt_messages, language=self.language)

            wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
            wx.CallAfter(self.chat_display.AppendText, f"{QA_res}")

            # answer the main_task_description question
            self.button_Disable()
            qa_gpt_response_main = QA_model.ask_gpt(main_task_description)
            wx.CallAfter(self.button_Enable, qa_gpt_response_main)

            if self.language == "zh":
                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI: 累计问题数为0  ==========\n")
            if self.language == "en":
                wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI: Accumulate question number is 0  ==========\n")

            i = 1
            while i > 0:
                wx.CallAfter(self.chat_display.AppendText, f"{QA_ask}")
                QA_user_input = self.userInputQueue.get()

                if QA_user_input == 'exit':
                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                    wx.CallAfter(self.chat_display.AppendText, f"{end_msg}")
                    break
                if i > 2 and QA_user_input == 'add':
                    recent_chat = QA_model.messages[-6:]
                    # TODO: add Data_increment_GPTModel Here
                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI  ==========\n")
                    wx.CallAfter(self.chat_display.AppendText, f"data augmentation finished")
                else:
                    self.button_Disable()
                    qa_gpt_response = QA_model.ask_gpt(QA_user_input)
                    wx.CallAfter(self.button_Enable, qa_gpt_response)

                if self.language == "zh":
                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI: 累计问题数为{i}  ==========\n")
                if self.language == "en":
                    wx.CallAfter(self.chat_display.AppendText, f"==========  Smarton AI: Accumulate question number is {i}  ==========\n")
                i += 1

        except Exception as e:
            self.chat_display.AppendText(f"An exception occurred: {e}\n")

    def generate_checkbox(self, subtask_names, user_input):
        user_input = user_input.replace("，", ",")
        selected_index = [int(index) for index in user_input.split(',')]
        selected_subtask_names = [subtask_names[i - 1] for i in selected_index]
        return selected_index, selected_subtask_names

    def button_Enable(self, response, reason=None):
        text = self.chat_display.GetValue()
        new_text = text.replace('Asking GPT, please wait for a minute ...', '')
        self.chat_display.SetValue(new_text)

        self.chat_display.AppendText(f"==========  Smarton AI  ==========\n")
        self.chat_display.AppendText(f"{response}\n")
        if reason is not None:
            self.chat_display.AppendText(f"{reason}\n")
        self.send_button.Enable()

    def button_Disable(self):
        wx.CallAfter(self.chat_display.AppendText, f'Asking GPT, please wait for a minute ...')
        self.send_button.Disable()
