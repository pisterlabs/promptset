import ctypes
import io
import os
from PIL import ImageGrab
import requests
import base64
import urllib.parse
import openai

def block_focus(hwnd):  # 阻止窗口获取焦点

    GWL_EXSTYLE = -20
    WS_EX_NOACTIVATE = 0x08000000
    WS_EX_LAYERED = 0x00080000
    ctypes.windll.user32.SetWindowLongA(
        hwnd, GWL_EXSTYLE, WS_EX_NOACTIVATE | WS_EX_LAYERED)


def bind_hotkey(hWnd, hotkey):
    user32 = ctypes.windll.user32
    control = {}
    for i, key in enumerate(hotkey):
        control[i] = key
        user32.RegisterHotKey(hWnd, key, 0, key)


def screenshot(bbox=None):
    if bbox == None:
        user32 = ctypes.windll.user32
        user32.OpenClipboard(0)
        user32.EmptyClipboard()
        user32.CloseClipboard()
        dllPath = os.path.join(os.getcwd(), 'screenshot.dll')
        os.system("Rundll32.exe %s, CameraWindow" % dllPath)
        img = ImageGrab.grabclipboard()
        if img == None:
            return (False, None)
        with io.BytesIO() as output:
            img.save(output, format='png')
            bytes = output.getvalue()
    else:
        img = ImageGrab.grab(bbox=bbox)
        with io.BytesIO() as output:
            img.save(output, format='png')
            bytes = output.getvalue()
    return (True, bytes)


class BaiduApi():
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.token = self.get_access_token()

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials",
                  "client_id": self.api_key, "client_secret": self.secret_key}
        return str(requests.post(url, params=params).json().get("access_token"))

    def accurate(self, bytes):
        url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token=" + self.token
        base64_data = base64.b64encode(bytes).decode("utf-8")
        payload = f'image={urllib.parse.quote(base64_data)}&detect_direction=false&paragraph=false&probability=false&language_type=auto_detect&detect_language=true'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()


class ChatGPT():
    def __init__(self, base_url, api_key):
        openai.api_base = base_url
        openai.api_key = api_key

    def send(self, prompt, content,callback=None,model="gpt-3.5-turbo",max_tk=1024):
        print("正在发送")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user", "content": content
                }
            ],
            max_tokens = max_tk,
            temperature=0,
            stream=True
        )
        print("响应中")
        text = []
        for item in response:
            msg = item['choices'][0]['delta']
            if item['choices'][0]['finish_reason'] != "stop":
                if "content" in msg:
                    text.append(msg.content)
                if callback!=None:
                    callback("".join(str(x) for x in text))
            else:
                print("")
        print("正在关闭")
        return "".join(str(x) for x in text)