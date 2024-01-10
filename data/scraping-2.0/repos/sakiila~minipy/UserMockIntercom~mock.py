import openai
import PySimpleGUI as sg

sg.theme('DarkTeal9')    # 设置主题

# OpenAI API Key
openai.api_key = "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"

# 设置使用OpenAI API的基本配置信息
model_engine = "text-davinci-002"  # GPT-3 的模型引擎
prompt_text = "你是一个使用 MoeGo 宠物预约系统的用户，你遇到一些使用上的问题，需要向我请教。"  # 开始对话的提示语
temperature = 0.9   # 扰动系数，用于让模型更具创造性的生成文本
max_tokens = 500     # 最大 tokens 数
stop_sequence = "\n"    # 对话结束的标志

# 打开窗口并返回一个视图对象
def open_window():
  layout = [
    [sg.Multiline("Intercom Chat\n", size=(70, 30), key='-OUTPUT-', font=('Helvetica', 14),
                  background_color='black', text_color='white')],
    [sg.Text('Input:'), sg.Input(key='-INPUT-', background_color='black', text_color='white')],
    [sg.Button('Send', bind_return_key=True, focus=True)]
  ]

  window = sg.Window('Intercom Chat', layout, resizable=True)

  return window

# 处理 API 请求函数
def get_response(prompt):
  try:
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n = 1,
        stop=stop_sequence
    )
  except Exception as e:
    sg.popup_quick_message("Something went wrong", font=('Helvetica', 14), background_color='black', text_color='white', auto_close_duration=2)
    return ""

  message = response.choices[0].text
  return str(message)

# 运行主程序
def main():
  # 打开窗口
  window = open_window()

  # 启动对话
  output_text = window['-OUTPUT-']
  input_text = window['-INPUT-']
  prompt = prompt_text

  ask = 1

  # 等待 API 的回复
  response = get_response(prompt)
  if response:
    ask = 0
    prompt += response

  # 对话循环
  while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
      break


    if ask == 0:
      output_text.update(f'Bot: {response}\n', append=True, text_color='lime')


    # 用户输入
    user_input = str(values['-INPUT-'])

    if event == 'Send' or (event == '\r' and user_input):
      # 如果用户点击发送按钮或者按下回车键，则向 API 发送请求
      output_text.update(f'You: {user_input}\n', append=True, text_color='white')

      # API 请求
      prompt = f"{prompt}\n{user_input}"
      response = get_response(prompt)

      # 等待 API 的回复
      if response:
        prompt += response
        output_text.update(f'Bot: {response}\n', append=True, text_color='lime')

      input_text.update('')   # 清空输入框

  window.close()

if __name__ == '__main__':
  main()
