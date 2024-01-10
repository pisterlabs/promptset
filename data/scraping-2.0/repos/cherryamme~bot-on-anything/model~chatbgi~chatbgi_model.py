# encoding:utf-8

from model.model import Model
from config import model_conf, common_conf_val
from common import const
from common.log import logger as log
import websocket
import openai
import time,json
import uuid,re

user_session = dict()
user_model = dict()
# OpenAI对话模型API (可用)
class ChatBGIModel(Model):
    def __init__(self):
        openai.api_key = model_conf(const.CHATBGI).get('api_key')
        openai.api_type = model_conf(const.CHATBGI).get('api_type')
        openai.api_version = model_conf(const.CHATBGI).get('api_version')
        api_base = model_conf(const.CHATBGI).get('api_base')
        self.model = model_conf(const.CHATBGI).get('model')
        self.function_list = model_conf(const.CHATBGI).get('function_list')
        if api_base:
            openai.api_base = api_base
        proxy = model_conf(const.OPEN_AI).get('proxy')
        if proxy:
            openai.proxy = proxy
        log.info("[CHATGPT] api_base={} proxy={}".format(
            api_base, proxy))
    def reply(self, query, context=None):
        query = query.strip()
        # acquire reply content
        function_list = self.function_list
        if not context or not context.get('type') or context.get('type') == 'TEXT':
            log.info("[CHATGPT] query={}".format(query))
            from_user_id = context['from_user_id']
            model = Session.return_model(from_user_id,self.model)
            # breakpoint()

            ###NOTE 指令收集#############
            help_command = common_conf_val('help_command', '#HELP')
            clear_memory_commands = common_conf_val('clear_memory_commands', ['#清除记忆'])
            change_gpt_mode = common_conf_val('change_gpt_mode',{"#GPT3":"gpt-3.5-turbo","#GPT4":"gpt-4"})
            search_command = common_conf_val('internet_search',['#SEARCH'])
            if query == help_command:
                return f"# 指令列表\n清空会话：{clear_memory_commands}\n切换模型：{change_gpt_mode}\n联网搜索：{search_command}\n帮助：{help_command}"
            if query in change_gpt_mode.keys():
                model=change_gpt_mode[query]
                Session.change_model(from_user_id,model)
                return f"切换模型:{model}"
            if query in clear_memory_commands:
                Session.clear_session(from_user_id)
                return '记忆已清除'
            # if query.startswith(tuple(search_command)):
            #     function_list = model_conf(const.CHATBGI).get('search_function_list')
            #     query = query.replace(search_command, "")
            # query,function_list = is_internet_search(query, function_list, search_command)
            ##NOTE 搜索不做替换
            if query.startswith(tuple(search_command)):
                function_list = model_conf(const.CHATBGI).get('search_function_list')
            elif query.startswith("#") and len(query) < 10:
                return f"指令无效,请使用 {help_command} 查看帮助"
            
            #############################

            conversation_id = Session.return_user_session(from_user_id)
            # conversation_id=Session
            query_json={"message":query, "conversation_id":conversation_id,"model":model,"function_list": function_list}
            new_query = json.dumps(query_json)
            log.info("[CHATGPT] session query={}".format(query_json))

            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, from_user_id)

            reply_content = self.reply_text(new_query, from_user_id, 0)
            #log.debug("[CHATGPT] new_query={}, user={}, reply_cont={}".format(new_query, from_user_id, reply_content))
            return reply_content

        elif context.get('type', None) == 'IMAGE_CREATE':
            return self.create_img(query, 0)

    def reply_text(self, query, user_id, retry_count=0):
        #DONE  更改这个函数，使其能够使用websockets
        #TODO  更改这个函数，使其能够将用户分流
        ws = websocket.WebSocket()
        ws.connect(url="ws://localhost:8000/conv", cookie="user_auth=wechat")
        # 将查询发送到 WebSocket 服务器
        ws.send(query)
        ## 参考更改为对话框格式
        def remove_html_tags(text):
            # 替换 <h4> 为换行符
            text = re.sub('<h4>', '\n', text)
            # 删除 </h4> 后的任意字符，直到遇到 <ol>
            pattern = re.compile(r'</h4>.*?<ol>', re.DOTALL)
            text = re.sub(pattern, '\n', text)
            # 替换 <li> 为破折号
            text = re.sub('<li>', '-', text)
            # 删除其他指定的HTML标签
            text = re.sub('<(\/?)(h[1-6]|ol|li)>', '', text)
            return text
        recv_message=""
        recv_conversation_id=""
        while True:
            try:
                response = ws.recv()
                ##TODO 修改这块，支持机器人回复参考资料
                recv_message = json.loads(response).get('message')
                ref_message = json.loads(response).get('reference')
                recv_conversation_id = json.loads(response).get('conversation_id')
                #保存recv_conversation_id
                Session.save_session(user_id, recv_conversation_id)
            except:
                break
        if isinstance(recv_message, str) and recv_message.startswith("正在搜索"):
            return "联网搜索繁忙，请稍后再试"
        log.info("[CHATGPT] response={}".format(recv_message))
        recv_message += remove_html_tags(ref_message)
        #保存recv_conversation_id
        return recv_message
        
    # async def reply_text_stream(self, query,  context, retry_count=0):
    #     #TODO  更改这个函数
    #     try:
    #         engine=model_conf(const.OPEN_AI).get("engine")
    #         print(f"{engine}")
    #         user_id=context['from_user_id']
    #         new_query = Session.build_session_query(query, user_id)
    #         res = openai.ChatCompletion.create(
    #             engine= model_conf(const.OPEN_AI).get("deployment") or "gpt-3.5-turbo",  # 对话模型的名称
    #             messages=new_query,
    #             temperature=model_conf(const.OPEN_AI).get("temperature", 0.75),  # 熵值，在[0,1]之间，越大表示选取的候选词越随机，回复越具有不确定性，建议和top_p参数二选一使用，创意性任务越大越好，精确性任务越小越好
    #             #max_tokens=4096,  # 回复最大的字符数，为输入和输出的总数
    #             #top_p=model_conf(const.OPEN_AI).get("top_p", 0.7),,  #候选词列表。0.7 意味着只考虑前70%候选词的标记，建议和temperature参数二选一使用
    #             frequency_penalty=model_conf(const.OPEN_AI).get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则越降低模型一行中的重复用词，更倾向于产生不同的内容
    #             presence_penalty=model_conf(const.OPEN_AI).get("presence_penalty", 1.0),  # [-2,2]之间，该值越大则越不受输入限制，将鼓励模型生成输入中不存在的新词，更倾向于产生不同的内容
    #             stream=True
    #         )
    #         full_response = ""
    #         for chunk in res:
    #             log.debug(chunk)
    #             if (chunk["choices"][0]["finish_reason"]=="stop"):
    #                 break
    #             chunk_message = chunk['choices'][0]['delta'].get("content")
    #             if(chunk_message):
    #                 full_response+=chunk_message
    #             yield False,full_response
    #         Session.save_session(query, full_response, user_id)
    #         log.info("[chatgpt]: reply={}", full_response)
    #         yield True,full_response

    #     except openai.error.RateLimitError as e:
    #         # rate limit exception
    #         log.warn(e)
    #         if retry_count < 1:
    #             time.sleep(5)
    #             log.warn("[CHATGPT] RateLimit exceed, 第{}次重试".format(retry_count+1))
    #             yield True, self.reply_text_stream(query, user_id, retry_count+1)
    #         else:
    #             yield True, "提问太快啦，请休息一下再问我吧"
    #     except openai.error.APIConnectionError as e:
    #         log.warn(e)
    #         log.warn("[CHATGPT] APIConnection failed")
    #         yield True, "我连接不到网络，请稍后重试"
    #     except openai.error.Timeout as e:
    #         log.warn(e)
    #         log.warn("[CHATGPT] Timeout")
    #         yield True, "我没有收到消息，请稍后重试"
    #     except Exception as e:
    #         # unknown exception
    #         log.exception(e)
    #         Session.clear_session(user_id)
    #         yield True, "请再问我一次吧"

    def create_img(self, query, retry_count=0):
        try:
            log.info("[OPEN_AI] image_query={}".format(query))
            response = openai.Image.create(
                prompt=query,    #图片描述
                n=1,             #每次生成图片的数量
                size="256x256"   #图片大小,可选有 256x256, 512x512, 1024x1024
            )
            image_url = response['data'][0]['url']
            log.info("[OPEN_AI] image_url={}".format(image_url))
            return [image_url]
        except openai.error.RateLimitError as e:
            log.warn(e)
            if retry_count < 1:
                time.sleep(5)
                log.warn("[OPEN_AI] ImgCreate RateLimit exceed, 第{}次重试".format(retry_count+1))
                return self.reply_text(query, retry_count+1)
            else:
                return "提问太快啦，请休息一下再问我吧"
        except Exception as e:
            log.exception(e)
            return None

class Session(object):
    @staticmethod
    def return_user_session(user_id):
        session = user_session.get(user_id,None)
        return session

    @staticmethod
    def save_session(user_id, recv_conversation_id):
        user_session[user_id] = recv_conversation_id
    @staticmethod
    def clear_session(user_id):
        user_session[user_id] = None
    @staticmethod
    def change_model(user_id,model_name):
        user_model[user_id] = model_name
    @staticmethod
    def return_model(user_id,default_model):
        model = user_model.get(user_id,default_model)
        return model
