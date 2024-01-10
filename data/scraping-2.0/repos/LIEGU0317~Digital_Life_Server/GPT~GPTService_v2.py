import json
import logging
import time

import httpx
import requests
from openai import OpenAI

from GPT import tune


class GPTService:
    def __init__(self, args):
        """
        GPTService
        初始化 GPTService 服务，设置相关参数。
        """
        self.history = []
        # when the reply gets cut off from hitting the maximum token limit (4,096 for gpt-3.5-turbo or 8,192 for gpt-4)
        if "4" in args.model:
            self.max_history = 8192
        else:
            self.max_history = 4096
        logging.info('初始化 ChatGPT 服务...')
        self.tune = tune.get_tune(args.character, args.model)  # 获取tune-催眠咒
        self.is_executed = False  # 标志变量，注入是否已经启用过，初始设置为 Fal
        self.model = args.model  # GPT 模型名称
        self.is_executed = False  # 标志变量，注入是否已经启用过，初始设置为 False
        # 设置代理
        proxies = {"http://": args.proxy, "https://": args.proxy} if args.proxy else None
        # defaults to os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(
            api_key=args.APIKey, http_client=httpx.Client(proxies=proxies), organization=None,
        )
        logging.info("ChatGPT 已初始化。")

    def add_to_history(self, user_input, gpt_response):
        # 添加用户输入和GPT响应到历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": gpt_response})
        # 如果历史记录太长，则截断
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
            logging.info('历史记录太长，截断。')

    def ask(self, text):
        """
        处理单轮请求。

        :param text: 输入文本。
        :return: GPT 模型的响应结果。
        """
        stime = time.time()
        try:
            if not self.is_executed:  # 如果不是第一次调用，则不需要在添加系统提示词
                self.is_executed = True  # 设置 is_executed 标志为 True
                messages = [{"role": "system", "content": f"{self.tune}"},
                            {"role": "user", "content": text}] + self.history
            else:
                messages = self.history + [{"role": "user", "content": text}]
            # 发送请求并获取响应
            completion = self.client.chat.completions.create(model=self.model, messages=messages, )
            # 提取回答内容
            message_content = completion.choices[0].message.content
            self.add_to_history(text, message_content)
            if message_content:
                logging.info('ChatGPT 响应：%s，用时 %.2f 秒' % (message_content, time.time() - stime))
                return message_content
            else:
                raise ValueError("Invalid response format")
        except json.JSONDecodeError as json_err:
            logging.error(f'JSON 解析失败，错误信息：{json_err}')
            return "ChatGPT-ask响应失败，返回的数据格式不正确。"
        except (requests.HTTPError, requests.Timeout) as req_err:
            logging.error(f'网络请求失败，错误信息：{req_err}')
            return "ChatGPT-ask响应失败，网络请求出错。"
        except Exception as e:
            logging.error('ChatGPT 响应失败，错误信息：%s' % e)
            return "GPT-ask响应失败，详情请及时联系系统管理员。"

    # 流式处理的逻辑可能需要根据 GPT 的特性进行适当调整
    def ask_stream(self, text):
        """
        处理流式请求。

        :param text: 输入文本。
        :yield: 流式响应的每个片段。
        """
        stime = time.time()
        try:
            if not self.is_executed:  # 如果不是第一次调用，则不需要再添加系统提示词
                self.is_executed = True  # 设置 is_executed 标志为 True
                messages = [{"role": "system", "content": f"{self.tune}"},
                            {"role": "user", "content": text}] + self.history
            else:
                messages = self.history + [{"role": "user", "content": text}]
            completion = self.client.chat.completions.create(model=self.model, messages=messages, stream=True)
            complete_text = ''
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    message_text = chunk.choices[0].delta.content
                    complete_text += message_text
                    # 检查是否形成完整句子
                    if any(punct in message_text for punct in ["。", "！", "？", "\n"]) and len(complete_text) > 3:
                        logging.info('ChatGPT 流式响应：%s，@时间 %.2f秒' % (complete_text, time.time() - stime))
                        yield complete_text.strip()
                        complete_text = ""
                if chunk.choices[0].finish_reason == "stop":
                    # 当收到结束标志时，如果有剩余文本，处理并返回
                    if complete_text:
                        logging.info('ChatGPT 流式响应（结束）：%s，@时间 %.2f秒' % (complete_text, time.time() - stime))
                        yield complete_text.strip()
                    break  # 退出循环

        except json.JSONDecodeError as json_err:
            logging.error(f'JSON 解析失败，错误信息：{json_err}')
            yield "ChatGPT-ask_stream 响应失败，返回的数据格式不正确。"

        except (requests.HTTPError, requests.Timeout) as req_err:
            logging.error(f'网络请求失败，错误信息：{req_err}')
            yield "ChatGPT-ask_stream 响应失败，网络请求出错。"

        except Exception as e:
            logging.error('ChatGPT 流式响应失败，错误信息：%s' % e)
            yield "GPT-ask_stream 响应失败，详情请及时联系系统管理员。"
