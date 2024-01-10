import os
import sqlite3
from bot import bot_factory
from bridge.bridge import Bridge
from bridge.context import ContextType
from channel.chat_message import ChatMessage
from config import conf
from plugins import *
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
import datetime
from common.log import logger
import plugins
import openai
from collections import Counter
from .lib import wxmsg as wx
import re
import google.generativeai as genai


@plugins.register(
    name="c_summary",
    desc="A plugin that summarize",
    version="0.1.0",
    author="cc",
    desire_priority=60
)


class ChatStatistics(Plugin):
    def __init__(self):
        super().__init__()

        # 设置数据库路径和API配置
        curdir = os.path.dirname(__file__)
        self.db_path = os.path.join(curdir, "chat.db")
        self.openai_api_key = conf().get("open_ai_api_key")
        self.openai_api_base = conf().get("open_ai_api_base", "https://api.openai.com/v1")
        self.gemini_api_key = conf().get("gemini_api_key")

        config_path = os.path.join(curdir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info(f"[c_summary] config content: {config}")
        self.ai_model = config.get("ai_model", "OpenAI")

        # 初始化数据库
        self.initialize_database()

        # 设置事件处理器
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        self.handlers[Event.ON_RECEIVE_MESSAGE] = self.on_receive_message

        # 记录初始化信息
        logger.info("[c_summary] Initialized")

    def initialize_database(self):
        """初始化数据库，创建所需表格和列"""
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                c = conn.cursor()
                # 创建表
                c.execute('''CREATE TABLE IF NOT EXISTS chat_records
                            (sessionid TEXT, msgid INTEGER, user TEXT, content TEXT, type TEXT, timestamp INTEGER, is_triggered INTEGER,
                            PRIMARY KEY (sessionid, msgid))''')
                
                # 检查并添加新列
                c.execute("PRAGMA table_info(chat_records);")
                if not any(column[1] == 'is_triggered' for column in c.fetchall()):
                    c.execute("ALTER TABLE chat_records ADD COLUMN is_triggered INTEGER DEFAULT 0;")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def _insert_record(self, session_id, msg_id, user, content, msg_type, timestamp, is_triggered=0):
        """向数据库中插入一条新记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO chat_records VALUES (?,?,?,?,?,?,?)", 
                          (session_id, msg_id, user, content, msg_type, timestamp, is_triggered))
            # logger.debug("insert chat record to db: %s", (session_id, msg_id, user, content, msg_type, timestamp, is_triggered))
        except Exception as e:
            logger.error(f"Error inserting record: {e}")

    def _get_records(self, session_id, excluded_users=None):
        """获取指定会话的当天聊天记录，排除特定用户列表中的用户"""
        if excluded_users is None:
            excluded_users = ["黄二狗²⁴⁶⁷","Oʀ ."]  # 默认排除的用户列表

        start_of_day = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_timestamp = int(start_of_day.timestamp())

        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()

                # 构建排除用户的 SQL 条件
                excluded_users_placeholder = ','.join('?' for _ in excluded_users)
                query = f"SELECT * FROM chat_records WHERE sessionid=? AND timestamp>=? AND user NOT IN ({excluded_users_placeholder}) ORDER BY timestamp DESC"

                # 准备查询参数
                query_params = [session_id, start_timestamp] + excluded_users

                # 执行查询
                c.execute(query, query_params)
                return c.fetchall()

        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            return []

    def on_receive_message(self, e_context: EventContext):
        context = e_context['context']
        cmsg : ChatMessage = e_context['context']['msg']
        username = None
        session_id = cmsg.from_user_id
        if conf().get('channel_type', 'wx') == 'wx' and cmsg.from_user_nickname is not None:
            session_id = cmsg.from_user_nickname # itchat channel id会变动，只好用群名作为session id
        # logger.debug(f"session_id: {session_id}")
        if context.get("isgroup", False):
            username = cmsg.actual_user_nickname
            if username is None:
                username = cmsg.actual_user_id
        else:
            username = cmsg.from_user_nickname
            if username is None:
                username = cmsg.from_user_id

        self._insert_record(session_id, cmsg.msg_id, username, context.content, str(context.type), cmsg.create_time)
        # logger.debug("[Summary] {}:{} ({})" .format(username, context.content, session_id))

# 在类中添加一个新的辅助方法
    def _get_session_id(self, chat_message: ChatMessage):
        session_id = chat_message.from_user_id
        if conf().get('channel_type', 'wx') == 'wx' and chat_message.from_user_nickname:
            session_id = chat_message.from_user_nickname
        return session_id

    def on_handle_context(self, e_context: EventContext):
        if e_context['context'].type != ContextType.TEXT:
            return

        content = e_context['context'].content
        chat_message: ChatMessage = e_context['context']['msg']
        # username = chat_message.actual_user_nickname or chat_message.from_user_id
        session_id = self._get_session_id(chat_message)
        prefix = "查群聊关键词"

        # 检查是否有切换模型的命令
        content_lower = content.lower()  # 将用户输入转换为小写

        # 检查是否有切换模型的命令
        if "cset openai" in content_lower:  # 使用转换后的小写字符串进行比较
            self.ai_model = "OpenAI"
            _set_reply_text("已切换到 OpenAI 模型。", e_context, level=ReplyType.TEXT)
            return
        elif "cset gemini" in content_lower:  # 使用转换后的小写字符串进行比较
            self.ai_model = "Gemini"
            _set_reply_text("已切换到 Gemini 模型。", e_context, level=ReplyType.TEXT)
            return

        # 解析用户请求
        elif "总结群聊" in content:
            logger.debug("开始总结群聊...")
            result = remove_markdown(self.summarize_group_chat(session_id, 100) ) # 总结最近100条群聊消息
            logger.debug("总结群聊结果: {}".format(result))
            _set_reply_text(result, e_context, level=ReplyType.TEXT)
            
        elif "群聊统计" in content:
            logger.debug("开始进行群聊统计...")
            ranking_results = self.get_chat_activity_ranking(session_id)
            logger.debug("群聊统计结果: {}".format(ranking_results))
            _set_reply_text(ranking_results, e_context, level=ReplyType.TEXT)
            
        elif content.startswith(prefix):
            # 直接提取关键词
            logger.debug("开始分析群聊关键词...")
            keyword = content[len(prefix):].strip()           
            if keyword:
                keyword_summary = remove_markdown(self.analyze_keyword_usage(keyword))
                _set_reply_text(keyword_summary, e_context, level=ReplyType.TEXT)
            else:
                _set_reply_text("请提供一个有效的关键词。", e_context, level=ReplyType.TEXT)

        elif content == "我的聊天":
            # 使用发送消息的用户昵称或用户ID
            user_identifier = chat_message.actual_user_nickname or chat_message.from_user_id
            if user_identifier:
                user_identifier = user_identifier.strip()
            logger.debug(f"开始分析用户{user_identifier}的聊天记录...")
            user_summary = remove_markdown(self.analyze_specific_user_usage(user_identifier))
            logger.debug(f"用户 {user_identifier} 的聊天记录分析结果: {user_summary}")
            _set_reply_text(user_summary, e_context, level=ReplyType.TEXT)


        else:
            # 使用正则表达式检查是否符合 "@xxx的聊天" 格式
            match = re.match(r"@([\w\s]+)的聊天$", content)
            if match:
                nickname = match.group(1).strip()
                logger.debug(f"开始分析群员{nickname}的聊天记录...")
                user_summary = remove_markdown(self.analyze_specific_user_usage(nickname))
                logger.debug(f"群员{nickname}的聊天记录分析结果: {user_summary}")
                _set_reply_text(user_summary, e_context, level=ReplyType.TEXT)
            else:
                e_context.action = EventAction.CONTINUE

    def _generate_model_analysis(self, prompt, combined_content):
        if self.ai_model == "OpenAI":
            messages = self._build_openai_messages(prompt, combined_content)
            return self._generate_summary_with_openai(messages)

        elif self.ai_model == "Gemini":
            messages = self._build_gemini_messages(prompt, combined_content)
            return self._generate_summary_with_gemini_pro(messages)

    def _build_openai_messages(self, prompt, user_input):
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]

    def _build_gemini_messages(self, prompt, user_input):
        prompt_parts = [
            prompt,
            "input: " + user_input,
            "output: "
        ]
        return prompt_parts

    def _generate_summary_with_openai(self, messages):
        """使用 OpenAI ChatGPT 生成总结"""
        try:
            # 设置 OpenAI API 密钥和基础 URL
            openai.api_key = self.openai_api_key
            openai.api_base = self.openai_api_base

            logger.debug(f"向 OpenAI 发送消息: {messages}")

            # 调用 OpenAI ChatGPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            logger.debug(f"来自 OpenAI 的回复: {json.dumps(response, ensure_ascii=False)}")
            reply_text = response["choices"][0]["message"]['content']  # 获取模型返回的消息
            return f"{reply_text}[O]"
        except Exception as e:
            logger.error(f"Error generating summary with OpenAI: {e}")
            return "生成总结时出错，请稍后再试。"


    def _generate_summary_with_gemini_pro(self, messages):
        """使用 Gemini Pro 生成总结"""
        try:
            # 配置 Gemini Pro API 密钥
            genai.configure(api_key=self.gemini_api_key)
            # Set up the model
            generation_config = {
            "temperature": 0.8,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8192,
            }

            # 创建 Gemini Pro 模型实例
            model = genai.GenerativeModel(model_name="gemini-pro",generation_config=generation_config)
            logger.debug(f"向 Gemini Pro 发送消息: {messages}")
            # 调用 Gemini Pro 生成内容
            response = model.generate_content(messages)
            reply_text = remove_markdown(response.text)
            logger.info(f"从 Gemini Pro 获取的回复: {reply_text}")
            return f"{reply_text}[G]"

        except Exception as e:
            logger.error(f"Error generating summary with Gemini Pro: {e}")
            return "生成总结时出错，请稍后再试。"

    def summarize_group_chat(self, session_id, count):
        # 从 _get_records 方法获取当天的所有聊天记录
        all_records = self._get_records(session_id)
        # 从所有记录中提取最新的 count 条记录，并只获取 user, content, timestamp 字段
        recent_records = [{"user": record[2], "content": record[3], "timestamp": record[5]} for record in all_records[:count]]
        logger.debug("recent_records: {}".format(recent_records))
        
        # 将所有聊天记录合并成一个字符串
        combined_content = "\n".join(
            f"[{datetime.datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}] {record['user']} said: {record['content']}"
            for record in recent_records
        )
        prompt = "你是一个群聊聊天记录分析总结助手，要根据获取到的聊天记录，将时间段内的聊天内容的主要信息提炼出来，适当使用emoji让生成的总结更生动。可以先用50字左右总结你认为最精华的聊天话题和内容。其次，对群聊聊天记录的内容要有深入的理解，可以适当提炼、分类你认为最精华的聊天主题，也可通过总结群聊记录来适当讨论群聊参与者的交互行为（总结的文本要连贯、排版要段落结构清晰、总体字数不超过150字。在总结的末尾单独一行，搭配emoji展示几个核心关键词（可以是活跃的群友名字、聊天数量、频次、主要话题等）,并进行一句话精华点评（搭配emoji)，约30字。"
        function_response = self._generate_model_analysis(prompt, combined_content)           
        logger.debug(f"Summary response from {self.ai_model}: {json.dumps(function_response, ensure_ascii=False)}")
        return function_response

    def get_chat_activity_ranking(self, session_id):
        """获取聊天活跃度排名前6位（当天）"""
        try:
            # 获取当天的聊天记录
            daily_records = self._get_records(session_id)
            # 使用 Counter 统计每个用户的消息数量
            user_message_count = Counter(record[2] for record in daily_records)
            # 根据消息数量排序
            sorted_users = user_message_count.most_common(6)
            # 获取排名第一的用户
            top_user = sorted_users[0][0] if sorted_users else None
            logger.debug(f"最活跃的用户: {top_user}")
            # 提取排名第一的用户的聊天内容
            top_user_messages = [record[3] for record in daily_records if record[2] == top_user]
            logger.debug(f"最活跃的用户的聊天内容: {top_user_messages[:5]}")
            # 如果有消息，将其发送给 Model
            if top_user_messages:
                # 构建消息格式
                formatted_top_user_messages = f"以下是 {top_user} 今天的聊天内容，请点评：\n" + "\n".join(top_user_messages)

                prompt = "你是一个群聊小助手，对获取到的群内最活跃的群员的聊天记录，进行适当的总结，并进行一句话点评（添加emoji)。总字数50字以内"
                messages_to_model = formatted_top_user_messages
                # 调用 Model 进行分析
                model_analysis = self._generate_model_analysis(prompt, messages_to_model)
                logger.debug(f"已完成群聊分析")
                # 处理 Model 的回复...
            # 生成排名信息
            ranking = ["😈 今日群员聊天榜🔝", "----------------"]  # 添加标题和分割线
            for idx, (user, count) in enumerate(sorted_users, start=1):
                emoji_number = self.get_fancy_emoji_for_number(idx)
                special_emoji = self.get_special_emoji_for_top_three(idx)
                ranking.append(f"{emoji_number} {user}: {count}条 {special_emoji}")
            logger.debug(f"活跃度排名成功: {ranking}")
            # 将 Model 的分析结果附加到排名信息之后
            final_result = "\n".join(ranking)
            if model_analysis:
                final_result += "\n\n🔍点评时刻:\n" + model_analysis
            return final_result
        except Exception as e:
            logger.error(f"Error getting chat activity ranking: {e}")
            return "Unable to retrieve chat activity ranking.", []


    def get_fancy_emoji_for_number(self, number):
        """为排名序号提供更漂亮的emoji"""
        fancy_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
        return fancy_emojis[number - 1] if number <= len(fancy_emojis) else "🔹"

    def get_special_emoji_for_top_three(self, rank):
        """为前三名提供特别的emoji"""
        special_emojis = ["✨", "🌟", "💫", "", "", ""]
        return special_emojis[rank - 1] if rank <= len(special_emojis) else ""

    def analyze_keyword_usage(self, keyword):
        # 调用 wxmsg 模块中的函数
        keyword_analysis = wx.analyze_keyword_in_messages(keyword)
        logger.debug(f"分析关键词 {keyword} 的使用情况成功: {keyword_analysis}")
        # 判断是否有有效的分析结果
        if keyword_analysis:
            # 准备 OpenAI 的输入
            messages_to_openai = [
                {"role": "system", "content": f"你是群里的聊天记录统计助手，你主要的功能是根据用户查询的关键词'{keyword}'，对和该关键词有关的聊天记录进行分析，形成一份简明、好看、完整的聊天记录报告，该报告要准确的结合聊天报告的文案风格，语言连贯，段落清晰，搭配数据加以展示。将获取到的聊天记录数据进行呈现，适当添加emoji，报告的角度包括但不限于该关键词讨论的热度、总提及次数、讨论最多的日期（频率、时间段）和该日提及次数、最多聊到该关键词的人是谁、聊了多少次....等等，以及根据提取出的特定聊天者针对该话题的聊天记录进行精彩点评。"},
                {"role": "user", "content": json.dumps(keyword_analysis, ensure_ascii=False)}
            ]
            # 调用 OpenAI 生成总结
            openai_analysis = self._generate_summary_with_openai(messages_to_openai)
            return openai_analysis
        else:
            return "没有找到关于此关键词的信息。"
    
    def analyze_specific_user_usage(self, nickname):
        # 调用 analyze_user_messages 函数进行分析
        user_analysis = wx.analyze_user_messages(nickname)
        logger.debug(f"分析用户{nickname}的使用情况: {user_analysis}")
        if user_analysis:
            # 准备 OpenAI 的输入
            messages_to_openai = [
                {"role": "system", "content": f"你是群里的聊天记录统计助手，主要的功能是分析群聊昵称名为【{nickname}】的聊天记录,精确整理出【{nickname}】的重要聊天信息。根据【{nickname}】的聊天记录各项数据生成一份专属于【{nickname}】的聊天记录报告，要求内容连贯、客观并体现数据，适当添加emoji使报告更美观，包括但不限于：各种类型的消息的发送数量、用户的消息最爱说哪些词汇、哪个时间段最爱聊天、该统计周期内总的聊天次数、聊天字数、话最多的一天是哪天（当天的发言条数和聊天字数）、用户的消息发送内容的情感倾向等等。报告要生动，对【{nickname}】和群员的互动进行精彩点评。"},
                {"role": "user", "content": user_analysis}
            ]

            # 调用 OpenAI 生成总结
            openai_analysis = self._generate_summary_with_openai(messages_to_openai)
            return openai_analysis
        else:
            return "没有找到关于此用户的信息。"


    def get_help_text(self, verbose=False, **kwargs):
        help_text = "一个清新易用的聊天记录统计插件。\n"
        if verbose:
            help_text += "使用方法: 总结群聊、聊天统计、聊天关键词等"
        return help_text
    

def _set_reply_text(content: str, e_context: EventContext, level: ReplyType = ReplyType.ERROR):
    reply = Reply(level, content)
    e_context["reply"] = reply
    e_context.action = EventAction.BREAK_PASS
# 其他必要的插件逻辑

def remove_markdown(text):
    # 替换Markdown的粗体标记
    text = text.replace("**", "")
    # 替换Markdown的标题标记
    text = text.replace("### ", "").replace("## ", "").replace("# ", "")
    return text