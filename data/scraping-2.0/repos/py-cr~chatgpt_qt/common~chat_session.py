# -*- coding:utf-8 -*-
# title           :chat_session.py
# description     :聊天会话
# author          :Python超人
# date            :2023-6-3
# link            :https://gitcode.net/pythoncr/
# python_version  :3.8
# ==============================================================================

# Define the ChatSession class  定义ChatSession类
from common.openai_chatbot import OpenAiChatbot


def build_i_have_asked(message):
    what_i_have_asked = {}
    what_i_have_asked["role"] = "user"
    what_i_have_asked["content"] = message
    return what_i_have_asked


def build_gpt_answer(message):
    what_gpt_answer = {}
    what_gpt_answer["role"] = "assistant"
    what_gpt_answer["content"] = message
    return what_gpt_answer


class ChatSession:
    # Define the create() class method to create a ChatSession object  定义create（）类方法，用于创建ChatSession对象
    @classmethod
    def create(cls):  # 类方法装饰器，用于调用方法时不需要实例化类
        # Return a ChatSession object  返回一个ChatSession对象
        return ChatSession()

    # Define the constructor with self parameter  定义构造函数，拥有self参数
    def __init__(self):
        # Initialize an empty list for chat history  初始化聊天记录的空列表
        self.chat_history = []
        # Set the system role as a string  将系统角色设为字符串
        self.system_role = "你是一位数学家"
        self.system_role = ""
        self.chatbot = OpenAiChatbot()

    # Define the send_message() method to send messages and get replies  定义send_message（）方法，用于发送消息并获取回复
    def send_message(self, message, model_id=None, api_key=None, proxy_enabled=None, proxy_server=None):
        """
        发送消息
        :param message: 消息
        :param model_id: openai 模型
        :param api_key: API Key
        :param proxy_enabled: "1"表示代理有效
        :param proxy_server: 代理服务器
        :return:
        """
        # Build the "I have asked" string  构建"I have asked"字符串
        i_have_asked = build_i_have_asked(message)

        # Get the current chat history list  获取当前聊天历史记录列表
        # chat_list = [item[1] for item in self.chat_history]

        # Check if system role is available, if so, add it to the chat history  检查系统角色是否可用，如果可用，则添加到聊天历史记录中
        if len(self.system_role) > 0:
            messages = [{"role": "system", "content": self.system_role}]
        else:
            messages = []

        # Append the "I have asked" message to chat history  将"I have asked"消息添加到聊天历史记录中
        self.chat_history.append((0, i_have_asked, len(message)))

        messages = messages + [item[1] for item in self.chat_history]

        content = ""
        error_count = 0
        # Get the reply from OpenAI  从OpenAI获得回复
        for reply, status, is_error in self.chatbot.chat_messages(messages,
                                                                  model_id=model_id,
                                                                  api_key=api_key,
                                                                  proxy_enabled=proxy_enabled,
                                                                  proxy_server=proxy_server):
            reply_content = reply["content"]
            content += reply_content
            if is_error == 1:
                error_count += 1

        # Call the AppEvents to update the chat history  调用AppEvents更新聊天历史记录
        # AppEvents.on_chat_history_changed()

        # Set the answer dict with the role and content keys  设置Answer字典，其中包含角色和内容键
        answer = {"role": "assistant", "content": content}

        # Append the answer to chat history  将Answer添加到聊天历史记录中
        self.chat_history.append((0, answer, len(content)))

        # Call the AppEvents to update the chat history  调用AppEvents更新聊天历史记录
        # AppEvents.on_chat_history_changed()

        # Return the answer content  返回Answer内容
        return content, error_count


# Check if the module is being run as the main program  检查模块是否作为主程序运行
if __name__ == '__main__':
    # Create a new ChatSession object  创建一个新的ChatSession对象
    session = ChatSession.create()

    # Send a message to the ChatSession object and get a reply  向ChatSession对象发送消息并获取回复
    reply = session.send_message("你好，1+1等于几？")

    # Send another message to the ChatSession object and get a reply  向ChatSession对象发送另一条消息并获取回复
    reply = session.send_message("我刚才问你什么问题？")

    # Print the second reply  打印第二个回复
    print("\nreply\n", reply)

    # Print the updated chat history  打印更新后的聊天历史记录
    print("\nchat_history\n", session.chat_history)
