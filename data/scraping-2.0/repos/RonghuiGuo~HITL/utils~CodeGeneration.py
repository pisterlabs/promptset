import os
import os.path as osp
import openai
import json
import re
import time
import dill
import cv2
import shutil
import time
import sqlite3

from pathlib import Path
from difflib import SequenceMatcher
from collections import namedtuple
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

class CodeGeneration():
    # #单例 保证只有一个实例
    # __instance = None
    # def __new__(cls):
    #     if cls.__instance is None:
    #         cls.__instance = super().__new__(cls)
    #     return cls.__instance

    def __init__(self):
        with open('config/default.json', 'r') as file:
            # Load the data from the file
            config_dict = json.load(file)
        Config = namedtuple('Config', config_dict.keys())
        args = Config(**config_dict)
        self.args = args
        openai.api_key = os.environ.get("openai_api_key")
        self.get_prompt()
        self.set_proxy()




    @staticmethod
    def set_proxy():
        # os.environ["http_proxy"] = "http://127.0.0.1:12345"
        # os.environ["https_proxy"] = "http://127.0.0.1:12345"
        # os.environ["ALL_PROXY"] = "http://127.0.0.1:12345"
        # os.environ["all_proxy"] = "http://127.0.0.1:12345"

        # os.environ["http_proxy"] = "http://127.0.0.1:12345"
        # os.environ["https_proxy"] = "http://127.0.0.1:12345"
        # os.environ["ALL_PROXY"] = "http://127.0.0.1:12345"
        # os.environ["all_proxy"] = "http://127.0.0.1:12345"

        # os.environ["http_proxy"] = "http://172.22.208.1:10809"
        # os.environ["https_proxy"] = "http://172.22.208.1:10809"
        # os.environ["http_proxy"] = "http://172.24.48.1:10809"
        # os.environ["https_proxy"] = "http://172.24.48.1:10809"
        pass

    def TopN_Feature2Scenarios(self, feature2scenarios_list, input_feature):


        similar_Feature2Scenarios = []
        # Define a function to calculate similarity score for a given feature

        for feature2scenarios in feature2scenarios_list:
            # Calculate the similarity between the input feature and the feature in the database
            similarity_score = SequenceMatcher(None, input_feature, feature2scenarios["feature"]).ratio()
            # If the similarity score is greater than or equal to 0.7, add the feature to the list
            if similarity_score >= self.args.similarity_threshold:
                similar_Feature2Scenarios.append({'feature': feature2scenarios["feature"], 'scenarios': feature2scenarios["scenarios"], 'similarity_score': similarity_score})

        # # Return the top similar features
        similar_Feature2Scenarios = sorted(similar_Feature2Scenarios, key=lambda x: x['similarity_score'], reverse=True)[:self.args.max_feature_number]
        return similar_Feature2Scenarios

    def get_prompt(self):
        with open(osp.join(self.args.prompt_path, "Gherkin_prompt.txt"), "r", encoding="utf-8") as f:
            self.Gherkin_prompt = f.read()
        with open(osp.join(self.args.prompt_path, "Design_page_prompt.txt"), "r", encoding="utf-8") as f:
            self.Design_page_prompt = f.read()
        with open(osp.join(self.args.prompt_path, "Visual_design_prompt.txt"), "r", encoding="utf-8") as f:
            self.Visual_design_prompt = f.read()
        with open(osp.join(self.args.prompt_path, "Code_generation_prompt.txt"), "r", encoding="utf-8") as f:
            self.Code_generation_prompt = f.read()

        with open(osp.join(self.args.prompt_path, "Gherkin2NL_prompt.txt"), "r", encoding="utf-8") as f:
            self.Gherkin2NL_prompt = f.read()
        with open(osp.join(self.args.prompt_path, "NL2Gherkin_prompt.txt"), "r", encoding="utf-8") as f:
            self.NL2Gherkin_prompt = f.read()
        with open(osp.join(self.args.prompt_path, "Gherkin_merge_prompt.txt"), "r", encoding="utf-8") as f:
            self.Gherkin_merge_prompt = f.read()

        with open(osp.join(self.args.prompt_path, "Code_modification_prompt.txt"), "r", encoding="utf-8") as f:
            self.Code_modification_prompt = f.read()

        with open(osp.join(self.args.prompt_path, "Test_cases_generation_prompt.txt"), "r", encoding="utf-8") as f:
            self.Test_cases_generation_prompt = f.read()

        with open(osp.join(self.args.prompt_path, "Code_modification_based_on_test_cases_prompt.txt"), "r", encoding="utf-8") as f:
            self.Code_modification_based_on_test_cases_prompt = f.read()

        with open(osp.join(self.args.prompt_path, "Human_in_the_loop_prompt.txt"), "r", encoding="utf-8") as f:
            self.Human_in_the_loop_prompt = f.read()

        with open(osp.join(self.args.prompt_path, "Design_modification_prompt.txt"), "r", encoding="utf-8") as f:
            self.Design_modification_prompt = f.read()

    def ask_chatgpt(self, messages):
        extra_response_count = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.args.temperature
                )
            except Exception as e:
                print(e)
                time.sleep(20)
                continue
            if response["choices"][0]["finish_reason"] == "stop":
                break
            else:
                messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
                messages.append({"role": "user", "content": "continue"})
                extra_response_count += 1
        return response, messages, extra_response_count

    def save_chat_messages(self, messages):
        with open(self.args.save_chat_path, "w", encoding="utf-8") as f:
            json.dump(messages, f)

    def save_code(self, code):
        with open(self.args.all_code_save_dir, "w", encoding="utf-8") as f:
            f.write(code)

    def Scenario_Parsing(self, Gherkin_response):
        # 用正则表达式进行解析
        gherkin_regex = re.compile(r'^\s*(?:Feature|Background|Scenario(?: Outline)?|Examples)\b')
        statements = []
        current_statement = ''
        for line in Gherkin_response.split('\n'):
            if gherkin_regex.match(line):
                if current_statement:
                    statements.append(current_statement.strip())
                    current_statement = ''
            current_statement += line + '\n'
        if current_statement:
            statements.append(current_statement.strip())

        Scenarios = []
        for i in range(len(statements)):
            if statements[i].startswith("Scenario"):
                Scenarios.append(statements[i])

        return Scenarios

    def Scenario_NL_Parsing(self, Scenario_NL):
        # 用正则表达式进行解析
        gherkin_regex = re.compile(r'^\s*(?:Feature|Background|Scenario(?: Outline)?|Examples)\b')
        statements = []
        current_statement = ''
        for line in Scenario_NL.split('\n'):
            if gherkin_regex.match(line):
                if current_statement:
                    statements.append(current_statement.strip())
                    current_statement = ''
            current_statement += line + '\n'
        if current_statement:
            statements.append(current_statement.strip())

        return statements

    def Gherkin_generation(self, input_feature, similar_Feature2Scenarios):
        Feature2Scenarios_str = ''
        if similar_Feature2Scenarios:
            for i, similar_Feature2Scenario in enumerate(similar_Feature2Scenarios):
                Feature2Scenarios_str = Feature2Scenarios_str+f"Feature {i}:"+similar_Feature2Scenario['feature']+"\n"
                for j, scenario in enumerate(similar_Feature2Scenario['scenarios']):
                    Feature2Scenarios_str = Feature2Scenarios_str+scenario+"\n"
                Feature2Scenarios_str = Feature2Scenarios_str+"\n"
            Human_in_the_loop_prompt = self.Human_in_the_loop_prompt.replace("{Replacement Flag}", Feature2Scenarios_str)
        else:
            Human_in_the_loop_prompt = ''

        messages = []
        Gherkin_prompt = self.Gherkin_prompt.replace("{Replacement Flag}", input_feature)
        Gherkin_prompt = Human_in_the_loop_prompt+Gherkin_prompt

        messages.append({"role": "user", "content": Gherkin_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        # 第一次要加上feature

        Gherkin_response = "Feature: "+input_feature+"\n"+"As a "
        Gherkin_response = self.handel_extra_response(extra_response_count, messages, Gherkin_response)
        # 得到最终的Gherkin_response
        Gherkin_response = Gherkin_response+response["choices"][0]["message"]["content"]

        return Gherkin_response, messages

    def Gherkin2NL(self, Scenarios_List, messages):
        Gherkin_NL_str = ''
        for i, scenario in enumerate(Scenarios_List):
            Gherkin_NL_str += scenario
            if i != len(Scenarios_List)-1:
                Gherkin_NL_str += "\n\n"

        Gherkin2NL_prompt = self.Gherkin2NL_prompt.replace("{Replacement Flag}", Gherkin_NL_str)
        # messages=[]
        messages.append({"role": "user", "content": Gherkin2NL_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        Gherkin_NL = ''
        Gherkin_NL = self.handel_extra_response(extra_response_count, messages, Gherkin_NL)
        Gherkin_NL = Gherkin_NL+response["choices"][0]["message"]["content"]

        Scenarios_NL_List = self.Scenario_NL_Parsing(Gherkin_NL)
        return Scenarios_NL_List

    def NL2Gherkin(self, Gherkin_NL_List, Feature):
        Gherkin_NL_str = ''
        for Gherkin_NL in Gherkin_NL_List:
            Gherkin_NL_str += Gherkin_NL+"\n"

        messages = []
        current_NL2Gherkin_prompt = self.NL2Gherkin_prompt.replace("{NL Replacement Flag}", Gherkin_NL_str)
        current_NL2Gherkin_prompt = current_NL2Gherkin_prompt.replace("{Feature Replacement Flag}", Feature)
        messages.append({"role": "user", "content": current_NL2Gherkin_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        Gherkin = ''
        Gherkin = self.handel_extra_response(extra_response_count, messages, Gherkin)
        Gherkin = Gherkin+response["choices"][0]["message"]["content"]
        Gherkin = "Feature:{Feature}\n".format(Feature=Feature)+Gherkin
        return Gherkin

    def Gherkin_merge(self, Gherkin_list):
        Gherkin_merge_str = ''  # 添加feature作为prompt
        for Gherkin in Gherkin_list:
            Gherkin_merge_str += Gherkin+"\n"
        Gherkin_merge_prompt = self.Gherkin_merge_prompt.replace("{Replacement Flag}", Gherkin_merge_str)
        messages = []
        messages.append({"role": "user", "content": Gherkin_merge_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        Gherkin_merge_results = ''
        Gherkin_merge_results = self.handel_extra_response(extra_response_count, messages, Gherkin_merge_results)
        Gherkin_merge_results = Gherkin_merge_results+response["choices"][0]["message"]["content"]
        return Gherkin_merge_results

    @staticmethod
    def handel_extra_response(extra_response_count, messages, response):
        if extra_response_count > 0:
            for i in range(extra_response_count):
                response += messages[(i-extra_response_count)*2]["content"]
        return response

    def Design_page_template_generation(self, Gherkin_Language):
        messages = []
        Design_page_template = ''

        Design_page_prompt = self.Design_page_prompt.replace("{Replacement Flag}", Gherkin_Language)
        messages.append({"role": "user", "content": Design_page_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        Design_page_template = self.handel_extra_response(extra_response_count, messages, Design_page_template)
        Design_page_template = Design_page_template+response["choices"][0]["message"]["content"]
        return Design_page_template

    def Visual_design_template_generation(self, Design_page_template):
        messages = []
        Visual_design_template = ''
        Visual_design_prompt = self.Visual_design_prompt.replace("{Replacement Flag}", Design_page_template)
        messages.append({"role": "user", "content": Visual_design_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        Visual_design_template = self.handel_extra_response(extra_response_count, messages, Visual_design_template)
        Visual_design_template = Visual_design_template+response["choices"][0]["message"]["content"]
        return Visual_design_template

    def Test_Cases_generation(self, Gherkin_result):
        messages = []
        Test_Cases = ''
        Test_cases_generation_prompt = self.Test_cases_generation_prompt.replace("{Replacement Flag}", Gherkin_result)
        messages.append({"role": "user", "content": Test_cases_generation_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        Test_Cases = self.handel_extra_response(extra_response_count, messages, Test_Cases)
        Test_Cases = Test_Cases+response["choices"][0]["message"]["content"]
        return Test_Cases

    def Code_modification_based_on_test_cases(self, Code, Test_Cases):
        messages = []
        Code_modification = ''
        Code_modification_based_on_test_cases_prompt = self.Code_modification_based_on_test_cases_prompt.replace("{Test Cases Replacement Flag}", Test_Cases)
        Code_modification_based_on_test_cases_prompt = Code_modification_based_on_test_cases_prompt.replace("{Code Replacement Flag}", Code)
        messages.append({"role": "user", "content": Code_modification_based_on_test_cases_prompt})
        response, messages, extra_response_count = self.ask_chatgpt(messages)
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        Code_modification = self.handel_extra_response(extra_response_count, messages, Code_modification)
        Code_modification = Code_modification+response["choices"][0]["message"]["content"]
        return Code_modification

    def Code_generation(self, Visual_design_template, Design_page_template, task, Gherkin_result):
        loop_number = 0
        while True:
            loop_number += 1
            messages = []
            Generate_code = ''
            Code_generation_prompt = self.Code_generation_prompt
            Code_generation_prompt = Code_generation_prompt.replace("{Visual_design_template Replacement Flag}", Visual_design_template)
            Code_generation_prompt = Code_generation_prompt.replace("{Design_page_template Replacement Flag}", Design_page_template)
            Code_generation_prompt = Code_generation_prompt.replace("{task Replacement Flag}", task)
            Code_generation_prompt = Code_generation_prompt.replace("{Gherkin_result Replacement Flag}", Gherkin_result)
            messages.append({"role": "user", "content": Code_generation_prompt})
            response, messages, extra_response_count = self.ask_chatgpt(messages)
            messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
            Generate_code = self.handel_extra_response(extra_response_count, messages, Generate_code)
            Generate_code = Generate_code+response["choices"][0]["message"]["content"]
            if self.Code_Parsing(Generate_code) or loop_number > self.args.max_retry:
                # # 生成测试用例
                # Test_Cases = self.Test_Cases_generation(Gherkin_result)
                # # 根据测试用例修改代码
                # Code_modification = self.Code_modification_based_on_test_cases(Generate_code,Test_Cases)
                # if self.Code_Parsing(Code_modification):
                #     return Code_modification,loop_number
                # else:
                #     self.logger.info("Code modification failed, please try again")
                #     continue
                return Generate_code, loop_number
            else:
                # self.logger.info("Code generation failed, please try again")
                continue

    def Replace_Images(self):

        png_placeholder = osp.join(self.args.static_dir, "img", 'Placeholder200.png')
        jpg_placeholder = osp.join(self.args.static_dir, "img", 'Placeholder200.jpg')

        with open(osp.join(self.args.static_html_dir, 'index.html')) as fp:
            html_soup = BeautifulSoup(fp, "html.parser")
        html_img_tags = html_soup.find_all("img")

        with open(osp.join(self.args.static_html_dir, 'style.css')) as fp:
            css_soup = BeautifulSoup(fp, "lxml")
        css_img_tags = css_soup.find_all("img")

        for img in html_img_tags:
            img_url = img.get("src")
            if not os.path.exists(osp.join(self.args.static_html_dir, img_url)):
                if img_url.endswith(".jpg"):
                    shutil.copyfile(jpg_placeholder, osp.join(self.args.static_html_dir, img_url))
                    # os.rename(jpg_placeholder, osp.join(self.args.static_html_dir, img_url))
                elif img_url.endswith(".png"):
                    # os.rename(png_placeholder, osp.join(self.args.static_html_dir, img_url))
                    shutil.copyfile(png_placeholder, osp.join(self.args.static_html_dir, img_url))
                else:
                    cv2.imwrite(osp.join(self.args.static_html_dir, img_url), cv2.imread(png_placeholder))

    # 使用正则表达式对生成的代码进行匹配，不一定准确

    def Code_Parsing(self, code):
        code_split = re.split("index.html:|style.css:|script.js:", code)
        try:
            static_html_dir=Path(self.args.static_html_dir)
            static_html_dir.mkdir(parents=True, exist_ok=True)
            with open(osp.join(self.args.static_html_dir, 'index.html'), 'w') as f:
                f.write(code_split[1])
            with open(osp.join(self.args.static_html_dir, 'style.css'), 'w') as f:
                f.write(code_split[2])
            with open(osp.join(self.args.static_html_dir, 'script.js'), 'w') as f:
                f.write(code_split[3])
            self.Replace_Images()
        except Exception as e:
            # self.logger.info(e)
            print(e)
            return False
        return True

    def Code_Modification(self, Generated_code, Code_Modification_String):
        loop_number = 0

        while True:
            loop_number += 1
            messages = []
            Modified_code = ''
            Code_modification_prompt = self.Code_modification_prompt.replace("{Code Replacement Flag}", Generated_code)
            Code_modification_prompt = Code_modification_prompt.replace("{Instructions Replacement Flag}", Code_Modification_String)
            messages.append({"role": "user", "content": Code_modification_prompt})
            response, messages, extra_response_count = self.ask_chatgpt(messages)
            messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
            Modified_code = self.handel_extra_response(extra_response_count, messages, Modified_code)
            Modified_code = Modified_code+response["choices"][0]["message"]["content"]

            if self.Code_Parsing(Modified_code) or loop_number > self.args.max_retry:
                # self.logger.info("Code modification completed!")
                return Modified_code, messages, loop_number
            else:
                # self.logger.info("Code modification failed, please try again")
                continue

    def Design_Modification(self, Generated_code, Code_Modification_String):
        loop_number = 0

        while True:
            loop_number += 1
            messages = []
            Modified_code = ''
            Design_modification_prompt = self.Design_modification_prompt.replace("{Code Replacement Flag}", Generated_code)
            Design_modification_prompt = Design_modification_prompt.replace("{Instructions Replacement Flag}", Code_Modification_String)
            messages.append({"role": "user", "content": Design_modification_prompt})
            response, messages, extra_response_count = self.ask_chatgpt(messages)
            messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
            Modified_code = self.handel_extra_response(extra_response_count, messages, Modified_code)
            Modified_code = Modified_code+response["choices"][0]["message"]["content"]

            if self.Code_Parsing(Modified_code) or loop_number > self.args.max_retry:
                # self.logger.info("Code modification completed!")
                return Modified_code, messages, loop_number
            else:
                # self.logger.info("Code modification failed, please try again")
                continue


    def clear_static_html_dir(self):
        static_html_dir = Path(self.args.static_html_dir)
        static_html_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(self.args.static_html_dir):
            os.remove(osp.join(self.args.static_html_dir, file))

            

    def copyfile2static_html_dir(self, origin_dir):
        for file in os.listdir(origin_dir):
            shutil.copyfile(osp.join(origin_dir, file), osp.join(self.args.static_html_dir, file))
