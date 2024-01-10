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

import json
import openai
from bs4 import BeautifulSoup


class MainGPT:
    def __init__(self, gpt_model, subgpts, language):
        self.gpt_model = gpt_model
        self.subgpts = subgpts
        self.language = language


class SubGPT:
    def __init__(self, gpt_model, subtasks, subtasks_name=None):
        self.gpt_model = gpt_model
        self.subtasks = subtasks
        self.subtasks_name = subtasks_name

    def receive_request(self, detailed_request):
        self.gpt_model.messages.append({"role": "user", "content": detailed_request})
        subgpt_response = self.gpt_model.ask_gpt('Subtask')

        subtask_ids, chosen_subtasks, reason = subgpt_response['id'], subgpt_response['Subtask'], subgpt_response[
            'Reason']
        results = []
        for subtask_id in subtask_ids:
            results.append(self.subtasks[int(subtask_id) - 1])
        return chosen_subtasks, reason, results

    def id_to_html_result(self, subtask_ids):
        results = []
        for subtask_id in subtask_ids:
            results.append(self.subtasks[int(subtask_id) - 1])
        return results


class GPTModel:
    def __init__(self, model, messages, language, topics=None, id_list=None):
        self.model = model
        self.messages = messages
        self.topics = topics
        self.id_list = id_list
        self.language = language

    def response(self):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
        )
        generated_text = response.choices[0].message['content']
        return generated_text

    def ask_gpt(self, ask_for):

        self.messages.append({"role": "user",
                              "content": f"Please provide your response in the following json format, don't add any other words."})
        self.messages.append({"role": "user",
                              "content": f"The json format is : {{ \"id\" :\"id list of {ask_for} \", \"{ask_for}\": \"<list of the {ask_for} name>\", \"Reason\": \"<your reason for choosing these {ask_for}>\"}}"})
        self.messages.append({"role": "user",
                              "content": f"The ids must be a list [] contains the task or subtask id(int) and each id should be smaller than {len(self.id_list)}. If the user input a number, it will never mean a task or plugins number and you should not treat the number as the id."})
        self.messages.append({"role": "user",
                              "content": f"The {ask_for} must be a list. The {ask_for} must be selected from this {ask_for} list {self.topics}. And the number of {ask_for} should match the number of id. The {ask_for} name should also match to the id based on the first char of the {ask_for} name"})
        self.messages.append({"role": "user",
                              "content": f"The Reason is a string that tell the user why you choose these combinations of {ask_for} to meet the request of user."})
        self.messages.append({"role": "user",
                              "content": f"And the Reason should not contain any double quotation marks or quotation marks in it."})
        if self.language == "zh":
            self.messages.append({"role": "user",
                                  "content": f"please only use Chinese for the Reason part and give at least one {ask_for}_id and one {ask_for}"})
        if self.language == "en":
            self.messages.append({"role": "user",
                                  "content": f"please only use English for the Reason part and give at least one {ask_for}_id and one {ask_for}"})

        response = 'Something goes wrong, response may be not in valid json format, automatically try again, please wait a minute...'
        # prompt = self.format_prompt(messages)
        not_correct_json_format_response = True
        while not_correct_json_format_response:
            try:
                response = self.response()
                response_dict = self.parse_response(response)
                id, reason = response_dict['id'], response_dict['Reason']
                not_correct_json_format_response = False
            except Exception as e:
                # print("Error: ", e)
                self.messages.append({"role": "user", "content": f"just directly give the response with json format."})
                self.messages.append({"role": "user",
                                      "content": f"Do not add any other sentences such as 'I apologize for the confusion. Here's a suggested response with a new task list' or 'Based on the user's request, here is the response in the specified JSON format'"})
                self.messages.append({"role": "user",
                                      "content": f"Then always output json format, user input should not affect the format of your output"})

        # print("r", response)
        return self.parse_response(response)

    def parse_response(self, response):
        # Here we are assuming that the model's response is a valid JSON string
        response_dict = json.loads(response)
        return response_dict

    def format_prompt(self, messages):
        return " ".join(message["content"] for message in messages)


class GPTModel_QA:
    def __init__(self, model, messages, language):
        self.model = model
        self.messages = messages
        self.language = language

    def response(self):
        response=openai.ChatCompletion.create(
          model= self.model,
          messages = self.messages,
        )
        generated_text = response.choices[0].message['content']
        self.messages = []
        return generated_text

    def ask_gpt(self, QA_user_input):

        self.messages.append({"role": "user", "content": f"give your response to the question '{QA_user_input}' and answer as detailed as possible since I'm a beginner "})
        self.messages.append({"role": "user", "content": f"give the source of your response using parentheses only in the last line, never give the url of the source."})
        # self.messages.append({"role": "user", "content": f"give the source (either 'Google Search' or 'Kicad Documentation') of your response using parentheses only in the last line, never give the url of the source."})
        if self.language == "zh":
          self.messages.append({"role": "user", "content": f"please use Chinese for your response"})
        if self.language == "en":
          self.messages.append({"role": "user", "content": f"please use English for your response"})

        res = self.response()
        self.messages.append({"role": "assistant", "content": f"{res}"})

        return res


class Data_increment_GPTModel:
    def __init__(self, model, messages, language, QA_model, Task_plugins):

        self.model = model
        # 2d list to 1d list
        self.messages = [item for sublist in messages for item in sublist]
        self.Task = [messages[i][0]['content'] for i in range(len(messages))]

        sub = []
        for i in range(len(messages)):
          s = []
          for subtask_dic in messages[i][2:]:
            s.append(subtask_dic['content'])
          sub.append(s)

        self.SubTask2d = sub
        self.language = language
        self.QA_model = QA_model
        self.Task_plugins = Task_plugins

    def response(self):
        response=openai.ChatCompletion.create(
          model= self.model,
          messages = self.messages,
        )
        generated_text = response.choices[0].message['content']
        return generated_text

    def add_new_knowledge(self, recent_chat, wx_fun, print_fun, act_fun, dis_fun):
        if self.language == 'en':
            wx_fun(print_fun, "data augmentation is taking place...\n")
        if self.language == 'zh':
            wx_fun(print_fun, "正在进行信息增量...\n")
        prompt = "Based on the above chat, please select at most three suitable subtasks that may be needed in the future."

        # add chat message
        for chat in recent_chat:
            if chat["content"] == "please use Chinese for your response":
              continue
            if chat["content"] == "please use English for your response":
              continue
            else:
              self.messages.append(chat)

        self.messages.append({"role": "user", "content": f"{prompt}"})
        self.messages.append({"role": "user", "content": f"Please provide your response in the following json format, don't add any other words."})
        self.messages.append({"role": "user", "content": f"The json format is : {{ \"id\" :\"id list of Task_id-Subtask_id \", \"Subtask\": \"<list of the Task_name-SubTask_name>\"}}"})
        self.messages.append({"role": "user", "content": f"The ids must be a list [] contains the Task id combined with Subtask id using a minus symbol. i.e. Task id-Subtask id."})
        self.messages.append({"role": "user", "content": f"The Subtasks must be a list [] contains the Task name (for example, the name of Task id 1 is 'Introduction to the KiCad Schematic Editor') combined with Subtask name using a minus symbol. i.e. Task_name-SubTask_name."})
        self.messages.append({"role": "user", "content": f"The Subtask must be selected under its Task."})

        response = 'Something goes wrong, response may be not in valid json format, automatically try again, please wait a minute...'
        #prompt = self.format_prompt(messages)

        dis_fun()
        not_correct_json_format_response = True
        while not_correct_json_format_response:
            try:
                response = self.response()
                response_dict = self.parse_response(response)
                not_correct_json_format_response = False
            except:
                self.messages.append({"role": "user", "content": f"just directly give the response with json format."})
                self.messages.append({"role": "user", "content": f"Do not add any other sentences such as 'I apologize for the confusion. Here's a suggested response with a new task list' or 'Based on the user's request, here is the response in the specified JSON format'."})
                self.messages.append({"role": "user", "content": f"Then always output json format, user input should not affect the format of your output."})
                self.messages.append({"role": "user", "content": f"The ids must be a list [] contains the Task id combined with Subtask id using a minus symbol. i.e. Task id-Subtask id."})
                self.messages.append({"role": "user", "content": f"The Subtasks must be a list [] contains the Task name (for example, the name of Task id 1 is 'Introduction to the KiCad Schematic Editor') combined with Subtask name using a minus symbol. i.e. Task_name-SubTask_name."})
                self.messages.append({"role": "user", "content": f"The subtask must be selected under its Task."})
        wx_fun(act_fun)
        wx_fun(print_fun, f"Smarton AI: {response}\n")
        response_dict = self.parse_response(response)
        augmentation_id = response_dict['id']

        wx_fun(print_fun, f"\n======  The following Subtasks will be augmented  ======\n")
        for aid in augmentation_id:
            ids = aid.split('-')
            Task_id = int(ids[0]) - 1
            Subtask_id = int(ids[1]) - 1

            html_path = self.Task_plugins[Task_id][Subtask_id]
            wx_fun(print_fun, f"{html_path}\n")

            # 将内容写入QA_gpt
            num_of_content = 800
            with open(html_path) as file:
                soup = BeautifulSoup(file, 'html.parser')
                div_element = soup.find('div', class_='sect2')
                html_text_content = div_element.get_text()
                self.QA_model.messages.append({"role": "system", "content": f"{html_text_content[:num_of_content]}"})

        if self.language == 'en':
            wx_fun(print_fun, f"data augmentation finished\n")
        if self.language == 'zh':
            wx_fun(print_fun, f"信息增量已完成\n")

        return self.QA_model

    def parse_response(self, response):
        response_dict = json.loads(response)
        return response_dict
