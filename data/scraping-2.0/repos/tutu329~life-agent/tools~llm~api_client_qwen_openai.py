import openai
# openai.api_base = "http://powerai.cc:8000/v1"
# openai.api_base = "http://localhost:8000/v1"
# openai.api_base = "http://127.0.0.1:8000/v1"
# openai.api_base = "http://116.62.63.204:8000/v1"
openai.api_key = "xxxxx"
from copy import deepcopy
import numpy as np
import os, requests
# import os, requests, torch
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.colors as mplc
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from PIL import Image
from typing import Collection, Dict, List, Set, Tuple, Union, Any, Callable, Optional
import random
import re

# ============================================关于qwen-vl中图片path的可访问性===============================================
# 1、本地文件系统方式【可行】，必须用反斜杠"\"，如：img_path = 'D:\\server\\static\\1.png'
# 2、远程方式【可行】，如：img_path = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
# 3、本地http链接方式【仍然卡死】，如：http://url:port/static/xxx.png，不清楚问题在哪里（可能是uvicorn的server配置问题）
# ===========（方式3：暂时不可行），要让客户端api请求中的'<img>localhost:8080/static/1.png</img> 图中内容有什么？'可访问============
# 方式3（暂时不可行）：需要在qwen-vl的openai_api.py的main中增加下面的代码，并新建D:/server/static，拷入图片：
# from fastapi.staticfiles import StaticFiles
# static_path = 'D:/server/static/'
# app.mount("/static", StaticFiles(directory=static_path), name="static")
# print(f'静态文件置于"{static_path}"中，外部访问：{args.server_name}:{args.server_port}/static/')
# ======================================================================================================================

# Qwen-VL目前不支持stream
# class LLM_Qwen_VL():
#     def __init__(self, temperature=0.7, url='http://127.0.0.1:8080/v1'):
#         self.url = url
#         self.temperature = temperature
#
#         self.images_path = []   # 图片链接list
#         self.images_info=''     # 图片链接的汇编string
#         self.images_index = 0   # 图片索引号
#
#         self.res = ''
#
#     def add_images(self, in_img_path_list):
#         self.images_path +=in_img_path_list
#
#         for img_path in in_img_path_list:
#             self.images_index += 1
#             self.images_info += f'pic{self.images_index}:' + f'<img>{img_path}</img>\n'
#
#     def clear_images(self):
#         self.images_path = []
#
#         self.images_info = ''
#         self.images_index = 0
#
#     def ask_block(self, in_query):
#         openai.api_base = self.url
#
#         query = ''
#         query += self.images_info
#         query += f'\t{in_query}'
#         print('User: \n\t', query)
#         res = openai.ChatCompletion.create(
#             model="Qwen",
#             temperature=self.temperature,
#             messages=[
#                 {"role": "user", "content": query},
#             ],
#             stream=False,
#             max_tokens=2048,
#         )
#         result = res['choices'][0]['message']['content']
#         print(f'Qwen-VL:\n\t{result}\n')
#         self.res = res
#         return res
#
#     # def _fetch_all_box_with_ref(self, text):
#     #     list_format = self.to_list_format(text)
#     #     output = []
#     #     for i, ele in enumerate(list_format):
#     #         if 'box' in ele:
#     #             bbox = tuple(map(int, ele['box'].replace('(', '').replace(')', '').split(',')))
#     #             assert len(bbox) == 4
#     #             output.append({'box': bbox})
#     #             if i > 0 and 'ref' in list_format[i-1]:
#     #                 output[-1]['ref'] = list_format[i-1]['ref'].strip()
#     #     return output
#
#     def get_boxes_info(self, s):
#         original_s = s
#         print(f'对象检测-解析字符串为：{s}')
#         def replace_last(target_string, replace_string, content):
#             i = content.rfind(target_string)
#             if i != -1:
#                 content = content[:i] + replace_string + content[i + len(target_string):]
#             return content
#
#         find_string = s
#
#         # 按批次获取ref和box，即必须解析完一个ref对应的所有box，才解析下一组ref及其box
#         find_index = 0
#         res_boxes = []
#
#         while s.count('<ref>')>0:
#             find_index += 1
#             print(f'=========================第{find_index}次循环===========================')
#             print(f'count of <ref> in "{s}": {s.count("<ref>")}')
#             if s.count('<ref>')>1:
#                 # 获取<ref></ref>与<ref></ref>之间的ref和box内容
#                 whole_ref_and_box_pattern = r'\<(ref)\>(.*?)\<(ref)\>'
#                 find_string = re.search(whole_ref_and_box_pattern, s)
#                 if find_string:
#                     find_string = find_string.group()
#                     find_string = replace_last('<ref>','', find_string)
#             elif s.count('<ref>')==1:
#                 # 获取<ref></ref>与结尾之间的ref和box内容
#                 whole_ref_and_box_pattern = r'\<(ref)\>(.*$)'
#                 find_string = re.search(whole_ref_and_box_pattern, s)
#                 if find_string:
#                     find_string = find_string.group()
#
#             print('find_string(获取<ref></ref>与<ref></ref>或结尾之间的ref和box内容)：', find_string)
#
#             s = s.replace(find_string, '')
#             # 本次循环完成后，删除解析部分即"<ref>狗头1</ref><box>(321,372),(373,460)</box>，<ref>狗头2</ref>" 中的 "<ref>狗头1</ref><box>(321,372),(373,460)</box>，"
#             print('删除find_string后的内容：', s)
#
#             # 解析find_string中所有<ref></ref>
#             pattern = r'\<(ref)\>(.*?)\</ref\>'
#             matches = re.findall(pattern, find_string)
#             refs = [{'ref': match[1]} for match in matches]
#
#             # 解析find_string中所有<box></box>
#             pattern = r'\<(box)\>(.*?)\</box\>'
#             matches = re.findall(pattern, find_string)
#             boxes = [{'box': eval(match[1])} for match in matches]
#
#             # 返回的boxes信息：
#             # [
#             #   {'ref':'some_obj', 'box':(x1, y1, x2, y2)  },
#             #   {'ref': '',        'box':(x1, y1, x2, y2)  },
#             # ]
#             ii = 0
#             for item in boxes:
#                 # append本次循环中find_string所包含的ref和box组成的box信息（其中，1个ref可能对应多个box）
#                 if len(refs) > ii:
#                     ref_value = refs[ii]['ref']
#                 else:
#                     ref_value = ''
#                 x1, y1 = boxes[ii]['box'][0]
#                 x2, y2 = boxes[ii]['box'][1]
#
#                 res_boxes.append(
#                     {'ref': ref_value, 'box': (x1, y1, x2, y2)},
#                 )
#                 ii += 1
#         print(f'对象检测-解析前完整字符串为：{original_s}')
#         print(f'对象检测-解析后的变量boxes为：{res_boxes}')
#         return res_boxes
#
#     def create_image_with_boxes(
#         self,
#         output_file_name,
#     ) -> Optional[Image.Image]:
#         image = self.images_path[-1]
#         # image = self._fetch_latest_picture(response, history)
#         if image is None:
#             return None
#         if image.startswith("http://") or image.startswith("https://"):
#             image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
#             h, w = image.height, image.width
#         else:
#             image = np.asarray(Image.open(image).convert("RGB"))
#             h, w = image.shape[0], image.shape[1]
#         visualizer = Visualizer(image)
#
#         # boxes = self._fetch_all_box_with_ref(response)
#         boxes = self.get_boxes_info(self.res['choices'][0]['message']['content'])
#         if not boxes:
#             return None
#         color = random.choice([_ for _ in mplc.TABLEAU_COLORS.keys()]) # init color
#         for box in boxes:
#             if 'ref' in box: # random new color for new refexps
#                 color = random.choice([_ for _ in mplc.TABLEAU_COLORS.keys()])
#             x1, y1, x2, y2 = box['box']
#             x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
#             visualizer.draw_box((x1, y1, x2, y2), alpha=1, edge_color=color)
#             if 'ref' in box:
#                 visualizer.draw_text(box['ref'], (x1, y1), color=color, horizontal_alignment="left")
#
#         image = visualizer.output
#         if image:
#             image.save(output_file_name)
#         else:
#             print("未检测到对象。")

# class VisImage:
#     def __init__(self, img, scale=1.0):
#         self.img = img
#         self.scale = scale
#         self.width, self.height = img.shape[1], img.shape[0]
#         self._setup_figure(img)
#
#     def _setup_figure(self, img):
#         fig = mplfigure.Figure(frameon=False)
#         self.dpi = fig.get_dpi()
#         # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
#         # (https://github.com/matplotlib/matplotlib/issues/15363)
#         fig.set_size_inches(
#             (self.width * self.scale + 1e-2) / self.dpi,
#             (self.height * self.scale + 1e-2) / self.dpi,
#         )
#         self.canvas = FigureCanvasAgg(fig)
#         # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
#         ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
#         ax.axis("off")
#         self.fig = fig
#         self.ax = ax
#         self.reset_image(img)
#
#     def reset_image(self, img):
#         img = img.astype("uint8")
#         self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")
#
#     def save(self, filepath):
#         self.fig.savefig(filepath)
#
#     def get_image(self):
#         canvas = self.canvas
#         s, (width, height) = canvas.print_to_buffer()
#
#         buffer = np.frombuffer(s, dtype="uint8")
#
#         img_rgba = buffer.reshape(height, width, 4)
#         rgb, alpha = np.split(img_rgba, [3], axis=2)
#         return rgb.astype("uint8")

# class Visualizer:
#     def __init__(self, img_rgb, metadata=None, scale=1.0):
#
#         if not os.path.exists("../../gpu_server/SimSun.ttf"):
#             ttf = requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/SimSun.ttf")
#             open("../../gpu_server/SimSun.ttf", "wb").write(ttf.content)
#         FONT_PATH = '../../gpu_server/SimSun.ttf'
#
#         self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
#         self.font_path = FONT_PATH
#         self.output = VisImage(self.img, scale=scale)
#         self.cpu_device = torch.device("cpu")
#
#         # too small texts are useless, therefore clamp to 14
#         self._default_font_size = max(
#             np.sqrt(self.output.height * self.output.width) // 30, 15 // scale
#         )
#
#     def draw_text(
#             self,
#             text,
#             position,
#             *,
#             font_size=None,
#             color="g",
#             horizontal_alignment="center",
#             rotation=0,
#     ):
#         if not font_size:
#             font_size = self._default_font_size
#
#         # since the text background is dark, we don't want the text to be dark
#         color = np.maximum(list(mplc.to_rgb(color)), 0.2)
#         color[np.argmax(color)] = max(0.8, np.max(color))
#
#         x, y = position
#         self.output.ax.text(
#             x,
#             y,
#             text,
#             size=font_size * self.output.scale,
#             fontproperties=FontProperties(fname=self.font_path),
#             bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
#             verticalalignment="top",
#             horizontalalignment=horizontal_alignment,
#             color=color,
#             zorder=10,
#             rotation=rotation,
#         )
#         return self.output
#
#     def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
#         x0, y0, x1, y1 = box_coord
#         width = x1 - x0
#         height = y1 - y0
#
#         linewidth = max(self._default_font_size / 4, 1)
#
#         self.output.ax.add_patch(
#             mpl.patches.Rectangle(
#                 (x0, y0),
#                 width,
#                 height,
#                 fill=False,
#                 edgecolor=edge_color,
#                 linewidth=linewidth * self.output.scale,
#                 alpha=alpha,
#                 linestyle=line_style,
#             )
#         )
#         return self.output
#
#     def get_output(self):
#         return self.output

class LLM_Qwen():
    def __init__(self, history=True, history_max_turns=50, history_clear_method='pop', temperature=0.7, url='http://127.0.0.1:8001/v1', need_print=True):
        self.url = url
        self.gen = None     # 返回结果的generator
        self.temperature = temperature
        print(f'------------------------------------------------------------------------------------------')
        print(f'LLM_Qwen(): temperature={self.temperature}')
        print(f'------------------------------------------------------------------------------------------')
        # self.top_k = top_k  # 需要回答稳定时，可以不通过调整temperature，直接把top_k设置为1; 官方表示qwen默认的top_k为0即不考虑top_k的影响

        # 记忆相关
        self.history_list = []
        self.history = history
        self.history_max_turns = history_max_turns
        self.history_turn_num_now = 0

        self.history_clear_method = history_clear_method     # 'clear' or 'pop'

        self.question_last_turn = ''
        self.answer_last_turn = ''

        self.role_prompt = ''
        self.has_role_prompt = False

        self.external_last_history = []     # 用于存放外部格式独特的history
        self.need_print = need_print

    # 动态修改role_prompt
    # def set_role_prompt(self, in_role_prompt):
    #     if in_role_prompt=='':
    #         return
    #
    #     self.role_prompt = in_role_prompt
    #     if self.history_list!=[]:
    #         self.history_list[0] = {"role": "user", "content": self.role_prompt}
    #         self.history_list[1] = {"role": "assistant", "content": '好的，我明白了，现在就开始，我会严格按照要求来。'}
    #     else:
    #         self.history_list.append({"role": "user", "content": self.role_prompt})
    #         self.history_list.append({"role": "assistant", "content": '好的，我明白了，现在就开始，我会严格按照要求来。'})

    def set_role_prompt(self, in_role_prompt):
        if in_role_prompt!='':
            # role_prompt有内容
            self.role_prompt = in_role_prompt
            if self.has_role_prompt and len(self.history_list)>0 :
                # 之前已经设置role_prompt
                self.history_list[0] = {"role": "user", "content": self.role_prompt}
                self.history_list[1] = {"role": "assistant", "content": '好的，我明白了，现在就开始，我会严格按照要求来。'}
            else:
                # 之前没有设置role_prompt
                self.history_list.insert(0, {"role": "user", "content": self.role_prompt})
                self.history_list.insert(1, {"role": "assistant", "content": '好的，我明白了，现在就开始，我会严格按照要求来。'})
                self.has_role_prompt = True
        else:
            # 删除role_prompt
            if self.has_role_prompt:
                if len(self.history_list)>0:
                    self.history_list.pop(0)
                if len(self.history_list)>0:
                    self.history_list.pop(0)
                self.has_role_prompt = False

    # 内部openai格式的history
    def __history_add_last_turn_msg(self):
        if self.history and self.question_last_turn != '':
            question = {"role": "user", "content": self.question_last_turn}
            answer = {"role": "assistant", "content": self.answer_last_turn}
            self.history_list.append(question)
            self.history_list.append(answer)
            if self.history_turn_num_now < self.history_max_turns:
                self.history_turn_num_now += 1
            else:
                if self.history_clear_method == 'pop':
                    print('======记忆超限，记录本轮对话、删除首轮对话======')
                    # for item in self.history_list:
                    #     print(item)
                    if self.role_prompt != '':
                        self.history_list.pop(2)
                        self.history_list.pop(2)
                    else:
                        self.history_list.pop(0)
                        self.history_list.pop(0)
                elif self.history_clear_method == 'clear':
                    print('======记忆超限，清空记忆======')
                    self.__history_clear()

    def clear_history(self):
        self.__history_clear()

    def __history_clear(self):
        self.history_list.clear()
        # self.has_role_prompt = False
        self.set_role_prompt(self.role_prompt)
        self.history_turn_num_now = 0

    # def __history_messages_with_question(self, in_question):
    #     msg_this_turn = {"role": "user", "content": in_question}
    #     if self.history:
    #         msgs = deepcopy(self.history_list)
    #         msgs.append(msg_this_turn)
    #         return msgs
    #     else:
    #         return [msg_this_turn]

    def __history_messages_with_question(self, in_question):
        msg_this_turn = {"role": "user", "content": in_question}
        msgs = deepcopy(self.history_list)
        msgs.append(msg_this_turn)
        return msgs

    def print_history(self):
        print('\n\t================对话历史================')
        for item in self.history_list:
            print(f"\t {item['role']}: {item['content']}")
        print('\t=======================================')

    # Undo: 删除上一轮对话
    def undo(self):
        if self.has_role_prompt:
            reserved_num = 2
        else:
            reserved_num = 0

        if len(self.history_list) >= reserved_num + 2:
            self.history_list.pop()
            self.history_list.pop()
            self.history_turn_num_now -= 1

        # if self.question_last_turn=='':
        #     # 多次undo
        #     if self.has_role_prompt:
        #         reserved_num = 2
        #     else:
        #         reserved_num = 0
        #
        #     if len(self.history_list)>=reserved_num+2:
        #         self.history_list.pop()
        #         self.history_list.pop()
        # else:
        #     # 一次undo
        #     self.question_last_turn=''

    def get_retry_generator(self):
        self.undo()
        return self.ask_prepare(self.question_last_turn).get_answer_generator()

        # temp_question_last_turn = self.question_last_turn
        # self.undo()
        # self.ask_prepare(temp_question_last_turn).get_answer_and_sync_print()

    # 返回stream(generator)
    def ask_prepare(
            self,
            in_question,
            in_temperature=0.7,
            in_max_new_tokens=2048,
            in_clear_history=False,
            in_stream=True,
            in_retry=False,
            in_undo=False,
            in_stop=None,
    ):
        # self.__history_add_last_turn_msg()

        if in_clear_history:
            self.__history_clear()

        if type(in_question)==str:
            # 输入仅为question字符串
            msgs = self.__history_messages_with_question(in_question)
        elif type(in_question)==list:
            # 输入为history list([{"role": "user", "content":'xxx'}, ...])
            msgs = in_question
        else:
            raise Exception('ask_prepare(): in_question must be str or list')

        # ==========================================================
        # print('发送到LLM的完整提示: ', msgs)
        print(f'------------------------------------------------------------------------------------------')
        print(f'ask_prepare(): temperature={self.temperature}')
        print(f'ask_prepare(): messages={msgs}')
        print(f'------------------------------------------------------------------------------------------')
        # ==========================================================

        if self.need_print:
            print('User: \n\t', msgs[-1]['content'])
        openai.api_base = self.url
        if in_stop is None:
            stop = ['</s>', 'human', 'Human', 'assistant', 'Assistant']
            # stop = ['</s>', '人类', 'human', 'Human', 'assistant', 'Assistant']
        else:
            stop = in_stop

        gen = openai.ChatCompletion.create(
            model="Qwen",
            temperature=in_temperature,
            # top_k=self.top_k,
            messages=msgs,
            stream=in_stream,
            max_new_tokens=in_max_new_tokens,   # 目前openai_api未实现（应该是靠models下的配置参数指定）
            # stop=stop,
            # Specifying stop words in streaming output format is not yet supported and is under development.
        )
        self.gen = gen

        self.question_last_turn = in_question
        return self

    def ask_block(self, in_question, in_clear_history=False, in_retry=False, in_undo=False):
        # self.__history_add_last_turn_msg()

        if in_clear_history:
            self.__history_clear()

        msgs = self.__history_messages_with_question(in_question)
        if self.need_print:
            print('User:\n\t', msgs[0]['content'])
        openai.api_base = self.url
        res = openai.ChatCompletion.create(
            model="Qwen",
            temperature=self.temperature,
            messages=msgs,
            stream=False,
            max_tokens=2048,
            functions=[
                {
                    'name':'run_code',
                    'parameters': {'type': 'object'}
                }
            ]
            # Specifying stop words in streaming output format is not yet supported and is under development.
        )
        result = res['choices'][0]['message']['content']
        if self.need_print:
            print(f'Qwen:\n\t{result}')
        return res

    # 方式1：直接输出结果
    def get_answer_and_sync_print(self):
        result = ''
        if self.need_print:
            print('Qwen: \n\t', end='')
        for chunk in self.gen:
            if hasattr(chunk.choices[0].delta, "content"):
                if self.need_print:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                result += chunk.choices[0].delta.content
                # yield chunk.choices[0].delta.content
        if self.need_print:
            print()
        self.answer_last_turn = result
        self.__history_add_last_turn_msg()

        return result

    # 方式2：返回generator，在合适的时候输出结果
    def get_answer_generator(self):
        answer = ''
        for chunk in self.gen:
            if hasattr(chunk.choices[0].delta, "content"):
                # print(chunk.choices[0].delta.content, end="", flush=True)
                answer += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content

        self.answer_last_turn = answer
        self.__history_add_last_turn_msg()

def main():
    llm = LLM_Qwen(history=True, history_max_turns=20, history_clear_method='pop')

    prompt = '不管发你什么，都直接翻译为英文，不解释。'
    llm.set_role_prompt(prompt)

    while True:
        question = input("user: ")
        # llm.ask(question).sync_print()
        for chunk in llm.ask_prepare(question).get_answer_generator():
            print(chunk, end='', flush=True)
        llm.print_history()

def main1():
    # from Util_Doc import *
    doc = Document('/Volumes/public/mbp15/mbp15_工作/===智慧能源====/200、===================科技项目===================/2023-08-07-LLM在能源电力系统咨询中的实战应用研究/南麂岛离网型微网示范工程-总报告.docx')
    table_content = []
    table = doc.tables[45]
    for i, row in enumerate(table.rows):
        text = [cell.text for cell in row.cells]
        table_content.append('\n'.join(text))
        print(text)
        # print(tuple(text))

    table_content = '\n'.join(table_content)
    print(table_content)

    llm = LLM_Qwen()
    question = f"你是电力系统专家，请总结这个表格'{table_content}' 的内容，并返回markdown格式的结果"
    print("user: ", question)
    print("Qwen: ", end='')
    llm.ask_prepare(question).get_answer_and_sync_print()

    # llm = LLM_Qwen()
    #
    # doc = Document('/Volumes/public/mbp15/mbp15_工作/===智慧能源====/200、===================科技项目===================/2023-08-07-LLM在能源电力系统咨询中的实战应用研究/南麂岛离网型微网示范工程-总报告.docx')
    # topic = '投资概算'
    # # topic = '建设规模'
    #
    # count_str = Text_Topic_Search(doc, topic).count()
    #
    # # question = f"请总结这段话：'{text}' 中关于建设规模的内容，去掉无关的内容"
    # question = f"你是电力系统专家，请总结这段话：'{count_str}' 中关于'{topic}'的内容，去掉与'{topic}'无关的内容，并返回markdown格式的结果"
    # print("user: ", question)
    # print("Qwen: ", end='')
    #
    # # -----------------------------直接输出-------------------------------
    # llm.ask(question).sync_print()






    # s_t = Epdi_Text()
    # s_t.init("/Volumes/public/mbp15/mbp15_工作/===智慧能源====/200、===================科技项目===================/2023-08-07-LLM在能源电力系统咨询中的实战应用研究/LLM测试文档.docx")
    # gen = s_t.get_paragraphs_generator_for_docx_file()
    # for para in gen:
    #     # print('段落: ', para)
    #     if '建设规模' in para:
    #         print("content: ", para)
    #
    #         # llm = LLM_Qwen()
    #         # text = para
    #         # background = '你正在协助我校核文档内容，你根据我的问题只以json格式数据的方式回复我，现在我开始提问题了，'
    #         # question = background + f"请问这段文字'{text}' " + "是否与电力工程建设规模相关（请注意，建设规模必须与主变台数、主变容量、间隔扩建情况、线路长度或线路截面之一有关）？如果与电力工程建设规模相关，绝对不要做任何解释，直接返回\{'sucess':True, 'content':text\}, 其中text为你对建设规模内容的总结文字；如果与电力工程建设规模无关，绝对不要做任何解释，直接返回\{'sucess':False, 'content':''\} "
    #         # llm.ask(question).sync_print()


    # ==============================================================================================================
    # llm = LLM_Qwen()
    #
    # text = '2.10.1 工程建设规模 1）新建薄刀咀光伏-沈家湾1回线，新建线路长度0.62km，采用截面为630mm2的电缆。2）扩建沈家湾变110kV间隔1个，进线电缆截面考虑630mm2。2.10.2 110kV主接线 本工程投产后，沈家湾变110kV母线接线维持不变。薄刀咀光伏电站110kV采用线变组接线。2.10.3 电气计算结论 薄刀咀光伏电站接入系统后，电网潮流分布合理，电压质量良好；电网发生故障且能正常切除的情况下，系统能够保持稳定，变电所各级电压满足规程规定；相关厂站短路电流均在其开关设备的额定遮断容量之内。'
    # question = f"请总结这段话：'{text}' 中，关于建设规模的内容"
    # # question = f"我正在校核报告内容，请问这段文字'{text}'，是否是关于电力工程建设规模的描述？"
    # print("user: ", question)
    #
    # print("Qwen: ", end='')
    #
    # # -----------------------------直接输出-------------------------------
    # llm.ask(question).sync_print()

    # -----------------------------获取gen输出-------------------------------
    # gen = llm.ask("你好").get_generator()
    # for chunk in gen:
    #     print(chunk, end='', flush=True)

    # ==============================================================================================================

def main9():
    import openai

    openai.api_key = "EMPTY"  # Not support yet
    # openai.api_key = "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"
    openai.api_base = "http://116.62.63.204:8000/v1"

    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    # 设置为本地的模型，因为vicuna使用的是假名字"text-embedding-ada-002"
    chat = ChatOpenAI(model="Qwen", temperature=0)
    answer = chat.predict_messages(
        [HumanMessage(content="Translate this sentence from English to Chinese. I love programming.")])
    print(answer)

def main():
    # print(f'openai.api_base: {openai.api_base}')
    # print(f'openai.api_key: {openai.api_key}')
    # print(f'openai.api_key_path: {openai.api_key_path}')
    # print(f'openai.api_version: {openai.api_version}')
    # print(f'openai.api_type: {openai.api_type}')

    llm = LLM_Qwen(
        history=True,
        history_max_turns=50,
        history_clear_method='pop',
        temperature=0.7,
        url='http://127.0.0.1:8001/v1'
    )
    llm.ask_prepare("写一个简单的markdown，里面有一个表格").get_answer_and_sync_print()

    # res = llm.ask_block('你是谁')
    # print(res['choices'][0]['message']['content'])

def main_vl():
    vl = LLM_Qwen_VL(temperature=0.51, url='http://127.0.0.1:8080/v1')
    vl.add_images([
        'D:\\server\\static\\1.jpeg',
        # 'D:\\server\\static\\1.png',
    ])
    res = vl.ask_block('输出狗头、狗爪、人脸和所有人手所在位置的检测框')
    # res = vl.ask_block('输出所有狗头和人手所在位置的检测框')
    # vl.clear_images()
    vl.create_image_with_boxes(output_file_name = 'D:\\server\\static\\box.jpg',)

if __name__ == "__main__" :
    main()
    # main_vl()

# create a request activating streaming response
# for chunk in openai.ChatCompletion.create(
#     model="Qwen",
#     messages=[
#         {"role": "user", "content": "在'游戏'、'看书'、'旅游'、'吃喝'、'玩乐'、'健身'、'思考'中随机选择一个"}
#     ],
#     max_tokens=1024,
#     stream=True,
#     # Specifying stop words in streaming output format is not yet supported and is under development.
# ):
#     if hasattr(chunk.choices[0].delta, "content"):
#         print(chunk.choices[0].delta.content, end="", flush=True)

# create a request not activating streaming response
# response = openai.ChatCompletion.create(
#     model="Qwen",
#     messages=[
#         {"role": "user", "content": "在'游戏'、'看书'、'旅游'、'吃喝'、'玩乐'、'健身'、'思考'中随机选择一个"}
#     ],
#     stream=False,
#     stop=[] # You can add custom stop words here, e.g., stop=["Observation:"] for ReAct prompting.
# )
# print(response.choices[0].message.content)

