import openai
import pyperclip
import time
import keyboard

openai.api_key="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"''
import keyboard
import pyperclip
# 创建一个空字符串用于储存输入
input_string = ''

def process_key(e):
    global input_string

    input_string += e.name

    if input_string.endswith(';;;'):
        print('Detected ;;;')

        # 使用shift+home键选中输入的字符串
        keyboard.send('shift+home')
        time.sleep(0.1)
        # 复制选中的字符串
        keyboard.send('ctrl+c')
        time.sleep(0.1)
        # 获取剪贴板中的内容
        text = pyperclip.paste()
        time.sleep(0.1)

        print(text)
        # 发送右键
        keyboard.send('right')
        # 发送三backspace
        keyboard.send('backspace')
        keyboard.send('backspace')
        keyboard.send('backspace')
        # 发送:

        pyperclip.copy(':')
        keyboard.send('ctrl+v')

        # 发送enter
        keyboard.send('enter')
        # 发送-
        keyboard.send('-')
        # 发送空格
        keyboard.send('space')

        response=openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0613",
          messages=[
                {"role": "system", "content": "用猫娘语气简要说明"},
                {"role": "user", "content": text}

            ]
        )
        # 获取response中choices的第一个元素的message的content
        print(response.choices[0].message.content)
        # 复制粘贴
        pyperclip.copy(response.choices[0].message.content)
        keyboard.send('ctrl+v')

        input_string = ''

keyboard.on_press(callback=process_key)

# 开始监听键盘输入
keyboard.wait()

