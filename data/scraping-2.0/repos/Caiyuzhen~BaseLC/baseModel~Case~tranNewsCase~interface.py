import gradio as gr

from langchain_helper import * # 🔥 * 为全部导入

with gr.Blocks() as demo:
    url = gr.Textbox()  # 让用户输入 url
    chatbot = gr.Chatbot() # gradio 的 chatbot 组件
    submit_btn = gr.Button("生成脱口秀剧本") # 提交按钮
    
    def generate_conversation(url):
        talkSHow: TalkShow = convertToTalkshow(url) # : 表示类型注解
        chat_history = [] # 把脱口秀的对话历史存进去
        
        # 拿到脱口秀的每一行, 每个角色说的话, 以 角色: XXX 的形式输出
        def parse_line(line: TalkShow_line): 
            if line is None:
                return ""
            return f'{line.character}: {line.content}'
            
        for i in range(0, len(talkSHow.script), 2): # 0 是 range 函数的起始值，表示循环将从列表的第一个元素（索引为 0）开始 | len(talkSHow.script) 是 range 函数的结束值 |  2 是 range 函数的步长(因为脱口秀的剧本返回的是元组 tuple),意味着循环将每次跳过一个元素。例如，如果 talkSHow.script 的长度为 6，那么 i 的值将依次是 0、2、4。这样可以确保在循环中每次处理一对台词（由两个不同的角色说出）
            line1 = talkSHow.script[i]
            line2 = talkSHow.script[i + 1] if (i+1) < len(talkSHow.script) else None # 👈表示循环次数 i < 脱口秀剧本的长度
            chat_history.append((parse_line(line1), parse_line(line2))) # 插入循环出来的【第一句话】跟【第二句话】 => 也就是两个脱口秀演员的对话
        return chat_history # 返回对话历史
        
	# 按钮的点击事件
    submit_btn.click(
		fn=generate_conversation, # 事件函数
		inputs=url, # 输入内容
		outputs=chatbot # 输出位置 => 把聊天记录显示到 chatbot => gradio 组件
	)
        
if __name__ == "__main__": # => python3 interface.py
    demo.launch() # 启动 gradio 组件