import re
import time
import logging

import openai
from openai.types.chat import ChatCompletion

from app.config.dev import *
from .prompt_manager import *

from app import the_redis as redis


class GptChatManager:
    def __init__(self, api_key: str = ""):
        """
        api_key: default in env
        """

        if api_key:
            self._api_key = api_key
        else:
            self._api_key = os.environ.get("OPENAI_API_KEY", "")
        assert self._api_key
        self._model_kind = DEFAULT_MODEL

        [getattr(self, method)() for method in dir(self) if
         callable(getattr(self, method)) and re.match(r'_new.*_chat', method)]

    def _new_general_chat(self):
        # 预热普通对话
        # 缺省网络连接配置：超时 3s, 重连 2 次
        self._general_chat = openai.OpenAI(api_key=self._api_key, timeout=3, max_retries=2).chat

    def _new_fault_desc_chat(self):
        # 预热对话
        # 网络连接配置：超时 3s, 重连 2 次
        self._fault_desc_chat = openai.OpenAI(api_key=self._api_key, timeout=3, max_retries=2).chat
        self._do_gpt_chat(self._fault_desc_chat, get_pre_hot_fault_desc(), 'system')

    def _new_expectation_chat(self):
        # 预热对话
        # 网络连接配置：超时 3s, 重连 2 次
        self._expectation_chat = openai.OpenAI(api_key=self._api_key, timeout=3, max_retries=2).chat
        self._do_gpt_chat(self._expectation_chat, get_pre_hot_expectation(), 'system')

    def _new_fault_report_chat(self):
        # 预热对话
        # 网络连接配置：超时 3s, 重连 2 次
        self._fault_report_chat = openai.OpenAI(api_key=self._api_key, timeout=3, max_retries=2).chat
        self._do_gpt_chat(self._fault_report_chat, get_pre_hot_fault_report(), 'system')

    def _new_advice_chat(self):
        # 预热对话
        # 网络连接配置：超时 3s, 重连 2 次
        self._advice_chat = openai.OpenAI(api_key=self._api_key, timeout=3, max_retries=2).chat
        self._do_gpt_chat(self._advice_chat, get_pre_hot_suggestion(), 'system')

    def _do_gpt_chat(self, chat, prompt: str, role: str = "user") -> (bool, ChatCompletion):
        """
        -> bool: valid resp
        -> ChatCompletion: ChatCompletion.Choice.message.content := str
        """
        n_retry = 3
        for i in range(n_retry):
            try:
                gpt_response = chat.completions.create(model=self._model_kind,
                                                       messages=[{"role": role,
                                                                  "content": prompt}],
                                                       timeout=60)
                return gpt_response.id is not None, gpt_response
            except openai.APITimeoutError as e:
                logging.error(f"[chatops]: {e}")
                if i < n_retry - 1:
                    # 等待
                    time.sleep(5)
                else:
                    return False, None
            except openai.RateLimitError as e:
                logging.error(f"[chatops]: {e}")
                return False, None
            except openai.APIStatusError as e:
                logging.error(f"[chatops]: {e}")
                return False, None

    def _do_gpt_chat_from_messages(self, chat, messages) -> (bool, ChatCompletion):
        """
        -> bool: valid resp
        -> ChatCompletion: ChatCompletion.Choice.message.content := str
        """
        try:
            gpt_response = chat.completions.create(model=self._model_kind,
                                                   messages=messages)
        except openai.APITimeoutError as e:
            logging.error(f"[chatops]: {e}")
            return False, None
        except openai.RateLimitError as e:
            logging.error(f"[chatops]: {e}")
            return False, None
        except openai.APIStatusError as e:
            logging.error(f"[chatops]: {e}")
            return False, None

        return gpt_response.id is not None, gpt_response

    def gpt_update_key(self, api_key: str) -> bool:
        """
        -> False: can't update api_key
        """
        self._api_key: str = api_key
        [getattr(self, method)() for method in dir(self) if
         callable(getattr(self, method)) and re.match(r'_new.*_chat', method)]
        return self.gpt_ping()

    def gpt_change_model(self, new_model: str) -> bool:
        """
        -> False: doesn't find new model
        """
        self._model_kind = new_model
        return self.gpt_ping()

    def gpt_ping(self) -> bool:
        prompt = "hi"
        success, _ = self._do_gpt_chat(self._general_chat, prompt)
        return success

    # 一
    def gen_fault_desc(self, fault_workflow: json, fault_config: json) -> (int, str):
        """
        fault_workflow:
        fault_config:
        """
        _, fault_desc_completion = self._do_gpt_chat(self._fault_desc_chat, get_prompt_fault_desc(fault_workflow))
        if not fault_desc_completion or len(fault_desc_completion.choices) == 0:
            return 0, None
        fault_desc = fault_desc_completion.choices[0].message.content

        # 将描述写回 redis
        fid = redis.incr('fid')
        redis.hset('faults', fid, fault_desc)

        return fid, fault_desc

    # 二
    def gen_expect(self, expectation_text: str) -> (int, json):
        """
        expectation_text:
        """
        # todo(cjx): 预处理用户期望文本
        # expectation_text = pre_handle(expectation_text)

        # 将期望写回 redis
        eid = redis.incr('eid')
        redis.hset('expectations', eid, expectation_text)

        _, expectation_completion = self._do_gpt_chat(self._expectation_chat, get_prompt_expectation(expectation_text))
        if not expectation_completion or len(expectation_completion) == 0:
            return 0, None
        expectation = expectation_completion.choices[0].message.content
        return eid, expectation

    # 三
    def gen_fault_result(self, fid) -> (int, str, str):
        """
        fid: fault id
        -> int: report id
        -> str: report content
        -> str: problem analysis content
        """
        fault = redis.hget('faults', fid).decode()
        if not fault:
            logging.error("fault no found")

        expectation = redis.hget('expectations', fid).decode()
        if not expectation:
            logging.error("expectation no found")

        # 生成 report
        _, fault_report_completion = self._do_gpt_chat_from_messages(self._fault_report_chat,
                                                                     get_messages_fault_report(fault,
                                                                                               expectation))

        if not fault_report_completion or len(fault_report_completion.choices) == 0:
            return 0, None, None

        # 将报告写回 redis
        rid = redis.incr('rid')
        fault_report = fault_report_completion.choices[0].message.content
        redis.hset('reports', rid, fault_report)

        # 生成 analysis
        _, fault_analysis_completion = self._do_gpt_chat(self._fault_report_chat, get_prompt_fault_analysis())

        if not fault_analysis_completion or len(fault_report_completion.choices) == 0:
            return 0, None, None
        fault_analysis = fault_analysis_completion[0]

        return rid, fault_report, fault_analysis

    # 四
    def gen_advice(self, fid) -> (int, str):
        """
        fid: fault id
        -> int: advice id
        -> str: advice content
        """
        fault = redis.hget('faults', fid).decode()
        if not fault:
            logging.error("fault no found")

        expectation = redis.hget('expectations', fid).decode()
        if not expectation:
            logging.error("expectation no found")

        report = redis.hget('reports', fid).decode()
        if not report:
            logging.error("report no found")

        _, advice_completion = self._do_gpt_chat_from_messages(self._fault_report_chat,
                                                               get_messages_suggestion(fault, expectation, report))

        if not advice_completion or len(advice_completion.choices) == 0:
            return 0, None

        # 将建议写回 redis
        aid = redis.incr('aid')
        advice = advice_completion.choices[0].message.content
        redis.hset('reports', aid, advice)

        return aid, advice
