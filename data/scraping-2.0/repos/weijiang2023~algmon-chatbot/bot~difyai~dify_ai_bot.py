# access DifyAI knowledge base platform
# docs: 

import json
import time
import requests

from bot.bot import Bot
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.openai.open_ai_image import OpenAIImage
from bot.session_manager import SessionManager
from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf

from dify_client import ChatClient


class DifyAIBot(Bot, OpenAIImage):
    # authentication failed
    AUTH_FAILED_CODE = 401
    NO_QUOTA_CODE = 406

    def __init__(self):
        super().__init__()
        self.sessions = SessionManager(ChatGPTSession, model=conf().get("model") or "gpt-3.5-turbo")

    def reply(self, query, context: Context = None) -> Reply:
        if context.type == ContextType.TEXT:
            return self._chat(query, context)
        elif context.type == ContextType.IMAGE_CREATE:
            ok, res = self.create_img(query, 0)
            if ok:
                reply = Reply(ReplyType.IMAGE_URL, res)
            else:
                reply = Reply(ReplyType.ERROR, res)
            return reply
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def _chat(self, query, context, retry_count=0) -> Reply:
        """
        发起对话请求
        :param query: 请求提示词
        :param context: 对话上下文
        :param retry_count: 当前递归重试次数
        :return: 回复
        """
        if retry_count >= 2:
            # exit from retry 2 times
            logger.warn("[DIFYAI] failed after maximum number of retry times")
            return Reply(ReplyType.ERROR, "请再问我一次吧")

        try:
            # load config
            difyai_api_key = conf().get("difyai_api_key")

            # Do http request by using the dify python sdk
            # Initialize ChatClient
            chat_client = ChatClient(difyai_api_key)

            # Create Chat Message using ChatClient
            # TODO: USE session_id, session and model?
            logger.info(f"[DIFYAI] query={query}")
            chat_response = chat_client.create_chat_message(inputs={}, query=query, response_mode="blocking", user="user_id")
            # TODO: if else branch based on the returned status
            chat_response.raise_for_status()
            #logger.info(f"[DIFYAI] status={status}")
            result = chat_response.text
            result = json.loads(result)
            reply_content = result.get('answer')
            logger.info(f"[DIFYAI] reply={reply_content}")
            # TODO: ADD self.sessions.session_reply(reply_content, session_id, total_tokens)
            return Reply(ReplyType.TEXT, reply_content)

        except Exception as e:
            logger.exception(e)
            # retry
            time.sleep(2)
            logger.warn(f"[DIFYAI] do retry, times={retry_count}")
            return self._chat(query, context, retry_count + 1)
