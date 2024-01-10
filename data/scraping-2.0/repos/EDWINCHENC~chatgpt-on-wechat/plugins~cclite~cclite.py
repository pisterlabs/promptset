import plugins
import openai
import requests
import json
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from channel.chat_message import ChatMessage
from config import conf
# from common.expired_dict import ExpiredDict
from plugins import *
from bridge import bridge
from bridge.bridge import Bridge
from common import const
from common.log import logger
from datetime import datetime
import os
import time
import traceback
import re
from .lib import fetch_tv_show_id as fetch_tv_show_id, tvshowinfo as tvinfo,function as fun,search_google as google,get_birth_info as birth, horoscope as horo
from .lib.model_factory import ModelGenerator


@plugins.register(
    name="cclite",
    desc="A plugin that supports multi-function_call",
    version="0.1.0",
    author="cc",
    desire_priority=66
)
class CCLite(Plugin):
    def __init__(self):
        super().__init__()
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        self.user_divinations = {}  # 将 user_divinations 作为类的属性
        curdir = os.path.dirname(__file__)
        config_path = os.path.join(curdir, "config.json")
        logger.info(f"[cclite] current directory: {curdir}")
        logger.info(f"加载配置文件: {config_path}")
        function_path = os.path.join(curdir, "lib", "functions.json")
        if not os.path.exists(config_path):
            logger.info('[RP] 配置文件不存在，将使用config.json.template模板')
            config_path = os.path.join(curdir, "config.json.template")
            logger.info(f"[cclite] config template path: {config_path}")
        try:
            with open(function_path, 'r', encoding="utf-8") as f:
                functions = json.load(f)
                self.functions = functions
                logger.info(f"[cclite] functions content: {functions}")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"[cclite] config content: {config}")
                self.openai_api_key = conf().get("open_ai_api_key")
                logger.info(f"[cclite] openai_api_key: {self.openai_api_key}")
                self.openai_api_base = conf().get("open_ai_api_base", "https://api.openai.com/v1")
                self.alapi_key = config["alapi_key"]    
                self.bing_subscription_key = config["bing_subscription_key"]
                self.google_api_key = config["google_api_key"]
                self.getwt_key = config["getwt_key"]
                self.cc_api_base = config.get("cc_api_base", "https://api.lfei.cc")
                self.google_cx_id = config["google_cx_id"]        
                self.functions_openai_model = config["functions_openai_model"]
                self.assistant_openai_model = config["assistant_openai_model"]
                self.temperature = config.get("temperature", 0.9)
                self.prompt = config.get("prompt", {})
                self.c_model = ModelGenerator()
                self.user_pets = {}  # 用于存储用户的宠物
                self.default_prompt = "当前中国北京时间是：{time}，你是一个可以通过联网工具获取各种实时信息、也可以使用联网工具访问指定URL内容的AI助手,请根据联网工具返回的信息按照用户的要求，告诉用户'{name}'想要的信息,要求排版美观，依据联网工具提供的内容进行描述！严禁胡编乱造！如果用户没有指定语言，默认中文。"
                logger.info("[cclite] inited")
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                logger.warn(f"[cclite] init failed, config.json not found.")
            else:
                logger.warn("[cclite] init failed." + str(e))
            raise e

    def get_prompt_for_function(self, function_name):
        return self.prompt.get(function_name, self.default_prompt)
    
    def base_url(self):
        return self.cc_api_base
    
    def on_handle_context(self, e_context: EventContext):
        context = e_context['context']
        msg: ChatMessage = context['msg']
        # user_id = msg.from_user_id
        isgroup = e_context["context"].get("isgroup")
        user_id = msg.actual_user_id if isgroup else msg.from_user_id
        nickname = msg.actual_user_nickname  # 获取nickname
        # 过滤不需要处理的内容类型
        if context.type not in [ContextType.TEXT, ContextType.IMAGE, ContextType.IMAGE_CREATE, ContextType.FILE, ContextType.SHARING]:
            # filter content no need solve
            return

        if context.type == ContextType.TEXT:

            content_lower = context.content.lower()
            if "cc openai" in content_lower:
                response = self.c_model.set_ai_model("OpenAI")
                _set_reply_text(response, e_context, level=ReplyType.TEXT)
                return
            elif "cc gemini" in content_lower:
                response = self.c_model.set_ai_model("Gemini")
                _set_reply_text(response, e_context, level=ReplyType.TEXT)
                return
            elif "cmodel" in content_lower:
                response = self.c_model.get_current_model()
                _set_reply_text(response, e_context, level=ReplyType.TEXT)
                return

            # 使用正则表达式来匹配星座运势的请求
            elif "运势" in context.content:
                match = re.search(r"(今日)?\s*(白羊座|金牛座|双子座|巨蟹座|狮子座|处女座|天秤座|天蝎座|射手座|摩羯座|水瓶座|双鱼座)\s*(运势|今日运势)?", context.content)
                if match:
                    sign = match.group(2)  # 获取匹配到的星座名称
                    logger.debug(f"正在获取 {sign} 星座运势数据")
                    _send_info(e_context, f"💰🧧 {sign}今日运势即将来临...")
                    try:
                        horoscope_data = horo.fetch_horoscope(sign)
                        logger.debug(f"星座运势响应：{horoscope_data}")
                        final_response = f"{horoscope_data}\n🔮 发送‘求签’, 让诸葛神数签诗为你今日算上一卦。"
                        _set_reply_text(final_response, e_context, level=ReplyType.TEXT)
                        return
                    except Exception as e:
                        logger.error(f"获取星座运势失败: {e}")
                        _set_reply_text(f"获取星座运势失败，请稍后再试", e_context, level=ReplyType.TEXT)
                        return

            elif "求签" in context.content:
                logger.debug("开始求签")
                # 检查用户是否已在当天抽过签
                if self.has_user_drawn_today(user_id):
                    response = "--今日已得签，请明日再来。--\n"
                    # 如果今日已求过签，显示今日的签文
                    if 'divination' in self.user_divinations[user_id]:
                        divination = self.user_divinations[user_id]['divination']
                        response += f"📜 今日{divination['qian']}"
                    _set_reply_text(response, e_context, level=ReplyType.TEXT)
                    return

                divination = horo.fetch_divination()
                if divination and divination['code'] == 200:
                    # 存储用户的抽签结果及日期
                    self.user_divinations[user_id] = {
                        'date': datetime.now().date().isoformat(),
                        'divination': divination,
                        'already_interpreted': False  # 初始化解签标记
                    }
                    logger.debug(f"当前抽签结果字典：{self.user_divinations}")
                    response = f"📜 你抽到了{divination['title']}\n⏰ {divination['time']}\n💬 {divination['qian']}\n🔮 发送‘解签’, 让诸葛神数为你解卦。"
                    _set_reply_text(response, e_context, level=ReplyType.TEXT)
                    return
                else:
                    _set_reply_text("获取签文失败，请稍后再试。", e_context, level=ReplyType.TEXT)
                    return

            elif "解签" in context.content:
                logger.debug("开始解签")
                # 检查用户是否已经抽过签
                if user_id in self.user_divinations and 'divination' in self.user_divinations[user_id]:
                    user_divination_data = self.user_divinations[user_id]
                    logger.debug(f"用户{user_id}的解签数据：{user_divination_data}")
                    # 检查是否已经解过签
                    if user_divination_data.get('already_interpreted', False):
                        _set_reply_text("今日已解签，请明日再来。", e_context, level=ReplyType.TEXT)
                        return
                    divination = user_divination_data['divination']
                    response = f"📖 {divination['jie']}"
                    _set_reply_text(response, e_context, level=ReplyType.TEXT)
                    # 标记为已解签
                    user_divination_data['already_interpreted'] = True
                    logger.debug(f"用户{user_id}已完成解签")
                    return
                else:
                    _set_reply_text("请先求签后再请求解签。", e_context, level=ReplyType.TEXT)
                    return

            elif re.search("吃什么|中午吃什么|晚饭吃什么|吃啥", context.content):
                logger.debug("正替你考虑今天吃什么")
                msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                nickname = msg.actual_user_nickname  # 获取nickname
                url = "https://zj.v.api.aa1.cn/api/eats/"
                try:
                    response = requests.get(url)
                    logger.debug(f"response响应：{response}")
                    if response.status_code == 200:
                        data = response.json()
                        logger.debug(f"data数据：{data}")
                        if data['code'] == 200:
                            # 构建消息
                            prompt = "你是中国著名的美食专家，走遍全国各大城市品尝过各种当地代表性的、小众的美食，对美食有深刻且独到的见解。你会基于背景信息，给用户随机推荐2道国内地域美食，会根据用户的烦恼给出合理的饮食建议和推荐的美食点评或推荐理由。"
                            user_input = f"对于今天该吃些什么好呢？你推荐了{data.get('meal1', '')}，和{data.get('meal2', '')}。现在需要你用两段文字（每段35字以内），适当结合{context.content}中用户的实际情况（例如来自什么地方、口味等）来简要点评推荐的菜、分享一下菜谱、营养搭配建议等，搭配适当的emoji来回复。总字数不超70字。"
                            # 调用OpenAI处理函数
                            model_response = self.c_model._generate_model_analysis(prompt, user_input)
                            # 构建最终的回复消息
                            final_response = (
                                f"🌟 你好呀！{nickname}，\n"
                                f"🍽️ 今天推荐给你的美食有：\n\n"
                                f"🍴 {data.get('meal1', '精选美食')} 或者 🍴 {data.get('meal2', '精选美食')}\n\n"
                                f"😊 奉上我的推荐理由：\n"
                                f"{model_response}"
                            )
                            logger.debug(f"_最终回复：{final_response}")
                            _set_reply_text(final_response, e_context, level=ReplyType.TEXT)
                            return
                except requests.RequestException as e:
                    return f"请求异常：{e}"

    #====================================================================================================
            #以下处理可能的函数调用逻辑
            input_messages = self.build_input_messages(context)
            logger.debug(f"Input messages: {input_messages}")

            # 运行会话并获取输出
            result = self.run_conversation(input_messages, e_context)
            called_function_name, conversation_output = result if result else (None, None)
            # 处理对话输出
            if conversation_output:
                # if called_function_name:
                # 处理当我们有一个具体的函数名时的情况
                # 如果函数返回的是视频播放源
                if called_function_name == "fetch_dyvideo_sources" and isinstance(conversation_output, list):
                    reply_type = ReplyType.VIDEO_URL
                    for video_url in conversation_output:
                        # 对于每个视频源，单独发送一个视频消息
                        _set_reply_text(video_url, e_context, level=reply_type)
                # else:
                #  ... 其他基于函数名称的逻辑 ...
                else:
                    # 对于其他类型的回复
                    conversation_output = remove_markdown(conversation_output)
                    reply_type = ReplyType.TEXT        
                    _set_reply_text(conversation_output, e_context, level=reply_type)

                logger.debug(f"Conversation output: {conversation_output}")


    def has_user_drawn_today(self, user_id):
        """检查用户是否在当天已求过签"""
        if user_id in self.user_divinations:
            last_divination_date = self.user_divinations[user_id].get('date')
            return last_divination_date == datetime.now().date().isoformat()
        return False

    def build_input_messages(self, context):
        find_content = context.content
        return [{"role": "user", "content": find_content}]

    def run_conversation(self, input_messages, e_context: EventContext):
        global function_response
        context = e_context['context']
        called_function_name = None  # 初始化变量
        messages = []
        openai.api_key = self.openai_api_key
        openai.api_base = self.openai_api_base        
        logger.debug(f"User input: {input_messages}")  #用户输入
        start_time = time.time()  # 开始计时
        response = openai.ChatCompletion.create(
            model=self.functions_openai_model,
            messages=input_messages,
            functions=self.functions,
            function_call="auto",
        )
        logger.debug(f"Initial response: {response}")  # 打印原始的response以及其类型
        message = response["choices"][0]["message"]  # 获取模型返回的消息。 
       
        logger.debug(f"message={message}")
        # 检查模型是否希望调用函数
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            called_function_name = function_name  # 更新变量
            logger.debug(f"Function call: {function_name}")  # 打印函数调用

            
            # 处理各种可能的函数调用，执行函数并获取函数的返回结果                       
            if function_name == "fetch_latest_news":  # 1.获取最新新闻
                api_url = f"{self.base_url()}/latest_news/"

                try:
                    # 发送GET请求到你的FastAPI服务
                    response = requests.get(api_url)
                    response.raise_for_status()  # 如果响应状态码不是200，将抛出异常
                    function_response = response.json()  # 解析JSON响应体为字典
                    logger.debug(f"Function response: {function_response}")  # 打印函数响应
                    function_response = function_response["results"]  # 返回结果字段中的数据
                    elapsed_time = time.time() - start_time  # 计算耗时
                    # 仅在成功获取数据后发送信息
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                        nickname = msg.actual_user_nickname  # 获取nickname
                        _send_info(e_context, f"@{nickname}\n✅获取实时要闻成功,正在整理。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅获取实时要闻成功,正在整理。🕒耗时{elapsed_time:.2f}秒")

                except requests.RequestException as e:
                    logger.error(f"Request to API failed: {e}")
                    _set_reply_text("获取最新新闻失败，请稍后再试。", e_context, level=ReplyType.TEXT)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return called_function_name, function_response
                
            elif function_name == "fetch_financial_news":  # 2.获取财经新闻
                api_url = f"{self.base_url()}/financial_news/"
                
                try:
                    # 发送GET请求到你的FastAPI服务
                    response = requests.get(api_url)
                    response.raise_for_status()  # 如果响应状态码不是200，将抛出异常
                    function_response = response.json()  # 解析JSON响应体为字典
                    logger.debug(f"Function response: {function_response}")  # 打印函数响应
                    function_response = function_response["results"]  # 返回结果字段中的数据
                    elapsed_time = time.time() - start_time  # 计算耗时
                    # 仅在成功获取数据后发送信息
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                        nickname = msg.actual_user_nickname  # 获取nickname
                        _send_info(e_context, f"@{nickname}\n✅获取实时财经资讯成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅获取实时财经资讯成功，正在整理。🕒耗时{elapsed_time:.2f}秒")

                except requests.RequestException as e:
                    logger.error(f"Request to API failed: {e}")
                    _set_reply_text("获取财经新闻失败，请稍后再试。", e_context, level=ReplyType.TEXT)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return called_function_name, function_response
            
            elif function_name == "get_weather_by_city_name":  # 3.获取天气
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                logger.debug(f"函数调用返回的原始字符串: {function_args_str}")
                function_args = json.loads(function_args_str)
                logger.debug(f"解析后的函数调用参数: {function_args}")
                city_name = function_args.get("city_name", "北京")  # 默认为北京
                logger.debug(f"查询的城市名参数值: {city_name} (类型: {type(city_name)})")
                adm = function_args.get("adm", None)  # 
                user_key = self.getwt_key

                if context.kwargs.get('isgroup'):
                    msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                    nickname = msg.actual_user_nickname  # 获取nickname
                    _send_info(e_context, "@{name}\n🔜正在获取{city}的天气情况🐳🐳🐳".format(name=nickname, city=city_name))
                else:
                    _send_info(e_context, "🔜正在获取{city}的天气情况🐳🐳🐳".format(city=city_name))

                # 向API端点发送GET请求，获取指定城市的天气情况
                try:
                    response = requests.get(
                        self.base_url() + "/weather/",
                        params={
                            "city_name": city_name,
                            "user_key": user_key,
                            "adm": adm
                        }
                    )
                    response.raise_for_status()  # 如果请求返回了失败的状态码，将抛出异常
                    function_response = response.json()
                    function_response = function_response.get("results", "未知错误")
                except Exception as e:
                    logger.error(f"Error fetching weather info: {e}")
                    _set_reply_text("获取天气信息失败，请稍后再试。", e_context, level=ReplyType.TEXT)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return called_function_name, function_response

            elif function_name == "request_train_info":  # 4.获取火车票信息
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)
                departure = function_args.get("departure", None)  # 默认值可以根据需要设置
                arrival = function_args.get("arrival", None)
                num_trains = function_args.get("num_trains", 3)  # 默认返回前3个车次
                date = function_args.get("date", None)  # 默认值为None，使用API的默认日期

                if context.kwargs.get('isgroup'):
                    msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                    nickname = msg.actual_user_nickname  # 获取nickname
                    _send_info(e_context, "@{name}\n🔍正在查询从{departure}到{arrival}的火车票信息，请稍后...".format(name=nickname, departure=departure, arrival=arrival))
                else:
                    _send_info(e_context, "🔍正在查询从{departure}到{arrival}的火车票信息，请稍后...".format(departure=departure, arrival=arrival))
                # 向端点发送请求，获取指定路线的火车票信息
                try:
                    response = requests.get(
                        self.base_url() + "/train_info/",
                        params={
                            "departure": departure,
                            "arrival": arrival,
                            "num_trains": num_trains,
                            "date": date
                        }
                    )
                    response.raise_for_status()  # 如果请求返回了失败的状态码，将抛出异常
                    function_response = response.json()
                    function_response = function_response['results']
                except Exception as e:
                    logger.error(f"Error fetching train info: {e}")
                    _set_reply_text("获取火车票信息失败，请稍后再试。", e_context, level=ReplyType.TEXT)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return function_response
                  
            elif function_name == "fetch_nowplaying_movies": # 6.获取正在上映的电影
                # 发送信息到用户，告知正在获取数据
                if e_context['context'].kwargs.get('isgroup'):
                    msg = e_context['context'].kwargs.get('msg')  # 这是WechatMessage实例
                    nickname = msg.actual_user_nickname  # 获取nickname
                    _send_info(e_context, f"@{nickname}\n🔜正在获取最新影讯🐳🐳🐳")
                else:
                    _send_info(e_context, "🔜正在获取最新影讯🐳🐳🐳")

                # 构建API请求的URL
                api_url = f"{self.base_url()}/now_playing_movies/"

                # 向FastAPI端点发送GET请求
                try:
                    response = requests.get(api_url)
                    response.raise_for_status()  # 检查请求是否成功

                    # 解析响应数据
                    data = response.json()
                    function_response = data.get('results')
                    status_msg = data.get('status')
                    elapsed_time = data.get('elapsed_time')

                    # 根据响应设置回复文本
                    if status_msg == '失败':
                        _set_reply_text(f"\n❌获取失败: {status_msg}", e_context, level=ReplyType.TEXT)
                    else:
                        _set_reply_text(f"\n✅获取成功，耗时: {elapsed_time:.2f}秒\n{function_response}", e_context, level=ReplyType.TEXT)
                except requests.HTTPError as http_err:
                    # 如果请求出错，则设置失败消息
                    _set_reply_text(f"\n❌HTTP请求错误: {http_err}", e_context, level=ReplyType.TEXT)
                logger.debug(f"Function response: {function_response}")

                
            elif function_name == "fetch_top_tv_shows":  # 7.获取豆瓣最热电视剧榜单              
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)
                limit = function_args.get("limit", 10)
                type_ = function_args.get("type", 'tv')  # 默认为电视剧
                if context.kwargs.get('isgroup'):
                    msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                    nickname = msg.actual_user_nickname  # 获取nickname
                    _send_info(e_context,"@{name}\n☑️正在为您查询豆瓣的最热影视剧榜单🐳🐳🐳".format(name=nickname)) 
                else:
                    _send_info(e_context, "☑️正在为您查询豆瓣的最热影视剧榜单，请稍后...") 
                # 调用函数，获取豆瓣最热电视剧榜单
                # 向API端点发送GET请求，获取最热影视剧榜单
                try:
                    response = requests.get(
                        self.base_url() + "/top_tv_shows/",
                        params={
                            "limit": limit,
                            "type": type_,
                        }
                    )
                    response.raise_for_status()  # 如果请求返回了失败的状态码，将抛出异常
                    function_response = response.json()
                    function_response = function_response.get("results", "未知错误")
                except Exception as e:
                    logger.error(f"Error fetching top TV shows info: {e}")
                    _set_reply_text("获取最热影视剧榜单失败，请稍后再试。", e_context, level=ReplyType.TEXT)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                

            elif function_name == "fetch_ai_news":  # 7.获取AI新闻
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)
                max_items = function_args.get("max_items", 6)
                try:
                    response = requests.get(
                        self.base_url() + "/ainews/",
                        params={"max_items": max_items}
                    )
                    response.raise_for_status()  # 如果请求返回了失败的状态码，将抛出异常
                except Exception as e:
                    logger.error(f"Error fetching AI news: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    _set_reply_text(f"获取AI新闻失败，请稍后再试。错误信息: {e}", e_context, level=ReplyType.TEXT)
                    return  # 终止后续代码执行

                try:
                    function_response = response.json()
                    function_response = function_response.get("results", "未知错误")
                except ValueError as e:  # 捕获JSON解析错误
                    logger.error(f"JSON parsing error: {e}")
                    function_response = "未知错误"

                elapsed_time = time.time() - start_time  # 计算耗时

                try:
                    # 发送信息
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                        nickname = msg.actual_user_nickname  # 获取nickname
                        _send_info(e_context, f"@{nickname}\n✅获取AI资讯成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅获取AI资讯成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                except Exception as e:
                    logger.error(f"Error sending response: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return called_function_name, function_response
                
            elif function_name == "fetch_dyvideo_sources":  # 抖音视频源获取
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)
                search_content = function_args.get("search_content", "")
                max_videos = function_args.get("max_videos", 1)
                try:
                    response = requests.get(
                        self.base_url() + "/dyvideo_sources/",
                        params={"search_content": search_content, "max_videos": max_videos}
                    )
                    response.raise_for_status()  # 如果请求返回了失败的状态码，将抛出异常
                except Exception as e:
                    logger.error(f"Error fetching Douyin video sources: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    _set_reply_text(f"获取抖音视频源失败，请稍后再试。错误信息: {e}", e_context, level=ReplyType.TEXT)
                    return  # 终止后续代码执行

                try:
                    function_response = response.json()
                    function_response = function_response.get("results", "未知错误")
                    elapsed_time = time.time() - start_time  # 计算耗时
                    # 仅在成功获取数据后发送信息
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                        nickname = msg.actual_user_nickname  # 获取nickname
                        _send_info(e_context, f"@{nickname}\n✅获取dy视频成功。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅获取dy视频成功。🕒耗时{elapsed_time:.2f}秒")
                except ValueError as e:  # 捕获JSON解析错误
                    logger.error(f"JSON parsing error: {e}")
                    function_response = "未知错误"
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return called_function_name, function_response

            elif function_name == "fetch_cls_news":  # 获取CLS新闻
                try:
                    response = requests.get(self.base_url() + "/clsnews/")
                    response.raise_for_status()
                except Exception as e:
                    logger.error(f"Error fetching CLS news: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    _set_reply_text(f"获取CLS新闻失败,请稍后再试,错误信息为 {e}", e_context, level=ReplyType.TEXT)

                try:
                    # 验证并解析JSON响应
                    function_response = response.json()
                except ValueError as e:  # 捕获JSON解析错误
                    logger.error(f"JSON parsing error: {e}")
                    function_response = "未知错误"
                else:
                    function_response = function_response.get("results", "未知错误")

                elapsed_time = time.time() - start_time  # 计算耗时

                # 发送信息
                try:
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')
                        nickname = msg.actual_user_nickname
                        _send_info(e_context, f"@{nickname}\n✅获取财联社新闻成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅获取财联社新闻成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                except Exception as e:
                    logger.error(f"Error sending response: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return called_function_name, function_response
                
                                       
            elif function_name == "fetch_hero_trending":  # 8.获取英雄热度趋势
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)
                hero_name = function_args.get("hero_name", "未指定英雄")
                if context.kwargs.get('isgroup'):
                    msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                    nickname = msg.actual_user_nickname  # 获取nickname
                    _send_info(e_context,"@{name}\n☑️正在为您进行指定英雄（{hero}）的数据获取，请稍后...".format(name=nickname, hero=hero_name)) 
                else:
                    _send_info(e_context, f"☑️正在为进行指定英雄（{hero_name}）的数据获取，请稍后...") 

                # 调用函数并获取返回值
                function_response = fun.get_hero_info(hero_name)
                # 转换为 JSON 格式
                # function_response = json.dumps(function_response, ensure_ascii=False)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                return called_function_name, function_response     
                
            elif function_name == "get_hero_ranking":  # 9.获取英雄梯度榜
                # 构建 API 请求的 URL
                api_url = f"{self.base_url()}/hero_ranking/"
                
                # 向 FastAPI 端点发送 GET 请求
                try:
                    response = requests.get(api_url)
                    response.raise_for_status()  # 检查请求是否成功
                    # 解析响应数据
                    data = response.json()
                    function_response = data.get('results')                    
                    # 根据响应设置回复文本
                    if function_response is None or "查询出错" in function_response:
                        _set_reply_text(f"❌获取失败: {function_response}", e_context, level=ReplyType.TEXT)
                    else:
                        _send_info(f"✅获取成功\n{function_response}", e_context, level=ReplyType.TEXT)
                except requests.HTTPError as http_err:
                    # 如果请求出错，则设置失败消息
                    _set_reply_text(f"❌HTTP请求错误: {http_err}", e_context, level=ReplyType.TEXT)
                except Exception as err:
                    # 如果发生其他错误，则设置失败消息
                    _set_reply_text(f"❌请求失败: {err}", e_context, level=ReplyType.TEXT)             
                # 记录响应
                logger.debug(f"Function response: {function_response}")
                                  
            elif function_name == "get_tv_show_interests":  # 10.获取电视剧或电影的评论
                com_reply = Reply()
                com_reply.type = ReplyType.TEXT
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)  # 使用 json.loads 将字符串转换为字典
                tv_show_name = function_args.get("tv_show_name", "未指定电视剧或电影")
                media_type = function_args.get("media_type", "tv")  # 默认为 'tv'
                count = function_args.get('count', 10)  # 默认10条评论
                order_by = function_args.get('orderBy', 'hot')  # 默认按照'hot'排序

                if context.kwargs.get('isgroup'):
                    msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                    nickname = msg.actual_user_nickname  # 获取nickname
                    _send_info(e_context,"@{name}\n☑️正在为您获取《{show}》的{media_type_text}信息和剧评，请稍后...".format(name=nickname, show=tv_show_name, media_type_text="电影" if media_type == "movie" else "电视剧")) 
                else:
                    _send_info(e_context,"☑️正在为您获取《{show}》的{media_type_text}信息和剧评，请稍后...".format(show=tv_show_name, media_type_text="电影" if media_type == "movie" else "电视剧")) 
                    
                # 使用 fetch_tv_show_id 获取电视剧 ID
                tv_show_id, status_msg, elapsed_time = fetch_tv_show_id.fetch_tv_show_id(tv_show_name)  # 假设函数返回 ID, 状态信息和耗时
                logger.debug(f"TV show ID: {tv_show_id}, status message: {status_msg}, elapsed time: {elapsed_time:.2f}秒")  # 打印获取的 ID 和状态信息                
                # 初始化回复内容
                com_reply.content = ""   # 假设 Reply 是一个您定义的类或数据结构
                
                # 根据获取的电视剧 ID 设置回复内容
                if tv_show_id is None:
                    # 如果获取 ID 失败，设置失败消息
                    com_reply.content += f"❌获取影视信息失败: {status_msg}"
                else:
                    # 如果获取 ID 成功，设置成功消息和链接
                    com_reply.content += f"✅获取影视信息成功，耗时: {elapsed_time:.2f}秒\n现可访问页面：https://m.douban.com/movie/subject/{tv_show_id}/\n以下为平台及播放跳转链接:"
                    
                    # 调用 fetch_media_details 函数获取影视详细信息
                    media_details = tvinfo.fetch_media_details(tv_show_name, media_type)
                    com_reply.content += f"\n{media_details}\n-----------------------------\n😈即将为你呈现精彩剧评🔜"  # 将详细信息添加到回复内容中
                    
                # 发送回复
                _send_info(e_context, com_reply.content)
                # 调用函数
                function_response = tvinfo.get_tv_show_interests(tv_show_name, media_type=media_type, count=count, order_by=order_by)  # 注意这里我们直接调用函数，并没有使用shows_map
                function_response = json.dumps({"response": function_response}, ensure_ascii=False)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应                
                
            elif function_name == "get_morning_news":  # 11.获取每日早报
                function_response = fun.get_morning_news(api_key=self.alapi_key)
                elapsed_time = time.time() - start_time  # 计算耗时
                _send_info(e_context, f"✅获取早报成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                
                                        
            elif function_name == "get_hotlist":      # 12.获取热榜信息
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)  # 使用 json.loads 将字符串转换为字典
                hotlist_type = function_args.get("type", "未指定类型")      
                try:
                    # 直接调用get_hotlist获取数据
                    function_response = fun.get_hotlist(api_key=self.alapi_key, type=hotlist_type)
                    logger.debug(f"Function response: {function_response}")
                except Exception as e:
                    logger.error(f"Error fetching hotlist: {e}")   
                    _set_reply_text(f"❌获取热榜信息失败,请稍后再试,错误信息为 {e}", e_context, level=ReplyType.TEXT)        
                    
            elif function_name == "bing_google_search":  # 13.搜索功能
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)  # 使用 json.loads 将字符串转换为字典
                search_query = function_args.get("query", "未指定关键词")
                search_count = function_args.get("count", 1)
                if "搜索" in context.content or "必应" in context.content.lower():
                    function_response = fun.search_bing(subscription_key=self.bing_subscription_key, query=search_query,
                                                        count=int(search_count))
                    function_response = json.dumps(function_response, ensure_ascii=False)
                    elapsed_time = time.time() - start_time  # 计算耗时
                    # 仅在成功获取数据后发送信息
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                        nickname = msg.actual_user_nickname  # 获取nickname
                        _send_info(e_context, f"@{nickname}\n✅Bing搜索{search_query}成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅Bing搜索{search_query}成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    logger.debug(f"Function response: {function_response}")  # 打印函数响应
                elif "谷歌" in context.content or "谷歌搜索" in context.content or "google" in context.content.lower():
                    function_response = google.search_google(search_terms=search_query, iterations=1, count=1,api_key=self.google_api_key, cx_id=self.google_cx_id,model=self.assistant_openai_model)
                    function_response = json.dumps(function_response, ensure_ascii=False)
                    elapsed_time = time.time() - start_time  # 计算耗时
                    # 仅在成功获取数据后发送信息
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                        nickname = msg.actual_user_nickname  # 获取nickname
                        _send_info(e_context, f"@{nickname}\n✅Google搜索{search_query}成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅Google搜索{search_query}成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    logger.debug(f"Function response: {function_response}")  # 打印函数响应
                else:
                    return None      

            elif function_name == "webpilot_search":  # 调用WebPilot内容获取函数
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)
                search_term = function_args.get("search_term", "")  # 默认搜索词为空字符串


                # 向API端点发送POST请求，获取与搜索词相关的内容
                try:
                    response = requests.post(
                        self.base_url() + "/webpilot_search/",
                        json={"search_term": search_term}
                    )
                    response.raise_for_status()  # 如果请求返回了失败的状态码，将抛出异常
                    function_response = response.json()
                    function_response = function_response.get("results", "未知错误")
                    elapsed_time = time.time() - start_time  # 计算耗时
                    # 仅在成功获取数据后发送信息
                    if context.kwargs.get('isgroup'):
                        msg = context.kwargs.get('msg')  # 这是WechatMessage实例
                        nickname = msg.actual_user_nickname  # 获取nickname
                        _send_info(e_context, f"@{nickname}\n✅Webpilot搜索{search_term}成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    else:
                        _send_info(e_context, f"✅Webpilot搜索{search_term}成功, 正在整理。🕒耗时{elapsed_time:.2f}秒")
                    logger.debug(f"Function response: {function_response}")  # 打印函数响应
                except Exception as e:
                    logger.error(f"Error fetching content: {e}")
                    _set_reply_text(f"获取内容失败，请稍后再试。错误信息 {e}", e_context, level=ReplyType.TEXT)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应

            elif function_name == "find_birthday":  # 查询生日信息
                # 从message里提取函数调用参数
                function_args_str = message["function_call"].get("arguments", "{}")
                function_args = json.loads(function_args_str)
                name = function_args.get("name", None)  # 如果没有提供名字，则默认查询最近的生日
                # 调用函数并获取返回值
                function_response = birth.find_birthday(name)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
                # return function_response

            elif function_name == "search_bing_news":  # 14.搜索新闻
                function_args = json.loads(message["function_call"].get("arguments", "{}"))
                logger.debug(f"Function arguments: {function_args}")  # 打印函数参数
                search_query = function_args.get("query", "未指定关键词")
                search_count = function_args.get("count", 10)
                function_response = fun.search_bing_news(count=search_count,subscription_key=self.bing_subscription_key,query=search_query, )
                function_response = json.dumps(function_response, ensure_ascii=False)
                logger.debug(f"Function response: {function_response}")  # 打印函数响应
            else:
                return                   
                    
                                               
            #以下为个性化提示词，并交给第二个模型处理二次响应
            prompt_template = self.get_prompt_for_function(function_name)

            msg: ChatMessage = e_context["context"]["msg"]
            current_date = datetime.now().strftime("%Y年%m月%d日%H时%M分")
            if e_context["context"]["isgroup"]:
                prompt = prompt_template.format(time=current_date, bot_name=msg.to_user_nickname,
                                                 name=msg.actual_user_nickname)
            else:
                prompt = prompt_template.format(time=current_date, bot_name=msg.to_user_nickname,
                                                 name=msg.from_user_nickname)
            # 将函数的返回结果发送给第二个模型
            logger.debug("messages: %s", [{"role": "system", "content": prompt}])
            # 打印即将发送给 openai.ChatCompletion 的 messages 参数
            logger.info("Preparing messages for openai.ChatCompletion...")
            logger.debug("messages: %s", [
                {"role": "system", "content": f"{prompt}"},*input_messages,
                {"role": "assistant", "content": f"{message}"},
                {"role": "function", "name": f"{function_name}", "content": f"{function_response}"}
            ])

            second_response = openai.ChatCompletion.create(
                model=self.assistant_openai_model,
                messages=[
                    {"role": "system", "content": f"{prompt}"},*input_messages,
                    {"role": "assistant", "content": f"{message}"},
                    {"role": "function", "name": f"{function_name}", "content":f"{function_response}"}
                ],
                temperature=float(self.temperature)
            )
            # 打印原始的second_response以及其类型
            second_response_json = json.dumps(second_response, ensure_ascii=False)
            logger.debug(f"Full second_response: {second_response_json}")
            logger.debug(f"called_function_name: {called_function_name}")
            # messages.append(second_response["choices"][0]["message"])
            return called_function_name, second_response['choices'][0]['message']['content']
        else:
            # 如果模型不希望调用函数，直接打印其响应
            logger.debug(f"模型未调用函数，原始模型响应: {message['content']}")  # 打印模型的响应
            return     

    def get_help_text(self, verbose=False, **kwargs):
        # 初始化帮助文本，插件的基础描述
        help_text = "\n🤖 基于微信的多功能聊天机器人，提供新闻、天气、火车票信息、娱乐内容等实用服务。\n"
        
        # 如果不需要详细说明，则直接返回帮助文本
        if not verbose:
            return help_text
        
        # 添加详细的使用方法到帮助文本中
        help_text += """
        🗞 实时新闻
        - "实时要闻", "实时新闻": 获取澎湃新闻的实时要闻。
        - "财经头条": 获取第一财经的财经新闻。可指定数量（默认8条）。
        - "AI资讯": 获取AI资讯。可指定数量（默认8条）。
        - "财联社新闻"：获取财联社新闻。可指定数量。

        🌅 每日早报
        - "早报": 获取每日早报信息。

        🔥 热榜信息
        - "热榜": 获取各大平台的热门话题。可指定平台（知乎、微博等）。

        🔍 搜索功能
        - "搜索 xxx": 使用必应、谷歌进行搜索，实现联网。
        - 新增 "Webpilot" 搜索功能，通过"w搜索"一般能准确调用，搜索内容质量较高。
"
        
        🎮 王者荣耀
        - "xx英雄数据", "xx英雄热度": 获取指定英雄的数据和趋势。
        - "英雄梯度榜": 获取王者荣耀英雄梯度榜。(来自苏苏的荣耀助手)

        📺 娱乐信息
        - "热播电视剧", "热播电影": 获取豆瓣热门电视剧和电影。
        - "热映电影": 获取电影院热映电影信息。
        - "电视剧/电影XXX": 获取指定电视剧或电影的信息、评价。

        ☀ 天气信息
        - "北京的天气怎么样": 获取北京的天气信息。
        
        🎥 抖音视频
        - "抖音+内容": 获取与搜索内容相关的抖音视频。
        """    
        # 返回帮助文本
        return help_text

def _send_info(e_context: EventContext, content: str):
    reply = Reply(ReplyType.TEXT, content)
    channel = e_context["channel"]
    channel.send(reply, e_context["context"])

def _set_reply_text(content: str, e_context: EventContext, level: ReplyType = ReplyType.ERROR):
    reply = Reply(level, content)
    e_context["reply"] = reply
    e_context.action = EventAction.BREAK_PASS

def remove_markdown(text):
    # 替换Markdown的粗体标记
    text = text.replace("**", "")
    # 替换Markdown的标题标记
    text = text.replace("### ", "").replace("## ", "").replace("# ", "")
    return text




