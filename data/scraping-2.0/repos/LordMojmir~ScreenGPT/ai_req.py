import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def query_custom_gpt(doc_text, message):
    """
    Sends a prompt to the GPT model and returns the response.

    :param doc_text: The document text to be analyzed.
    :param system_message: The system instruction for the GPT model.
    :return: The response text from the model.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    system_message = f"You are a helpful AI-assistant that will help the user with this {doc_text}"

    chat = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)



def convert_json_to_object(json_string):
    """
    Converts a JSON string to a Python dictionary.

    :param json_string: The JSON string to be converted.
    :return: A Python dictionary representing the JSON data.
    """
    try:
        python_object = json.loads(json_string)
        return python_object
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {str(e)}"

def obj2excel(data_list: object, excel_file_path: str) -> bool:
    df = pd.json_normalize(data_list)

    return df.to_excel(excel_file_path, index=False)


def save_json_to_file(json_obj, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(json_obj, json_file, indent=4)
        print(f"JSON data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to file: {e}")
#
# screen_content =  """File Edit View Navigate Code Refactor Run Iools Git Window Help ScreenGPT - main py C:IWindowsIsystem32 cmde: ScreenGPT mainpy Current File 0 Git: 2 Project 0 z * | & main py openai py (OCR2) C:|Userslhorvazpip install customtkinter ScreenGPT €: Userslhorval Documents| Pri 4 3 A 6 Collecting customtkinter env 48 submit_button ctk.CTkButton(root , text="Submit" command-Lambda: on_supmitler [ Downloading customtkinter-5.2.1-py3-none-any.whl metadata (652 bytes) gitignore 49 submit_button.pack(pady-10) Collecting darkdetect (from customtkinter) J main py DownLoading darkdetect-0 8.0-py3-none-any.whl (9 0 kB) openaipy 50 Downloading customtkinter-5.2.1-py3-none-any.whl (295 kB) READMEmd 296.0/296 0 kB 4.5 MB/s eta 0:00:00 screenshot png 51 root.mainZoop() 8 Installing collected packages = darkdetect customtkinter ] External Libraries 52 Successfully installed customtkinter-5.2.1 darkdetect-0.8.0 Scratches and Consoles 8 Mojmir Horvath (OCR2) C: |Userslhorva pip install pynput M 53 Idef for_canonical(f): { Collecting pynput Downl oading pynput-1.7.6-py2.py3-none-any.whl (89 kB) 54 listener keyboard.Listener(on_press-Lambda k: k) 89.2/89.2 kB 1.3 MB/s eta 0:00:00 Requirement already satisfied: six in c:luserslhorva anaconda3 lenvslocr2|liblsite-packages (from pynput) (1.16.0) 55 return Lambda k: f(listener.canonical(k)_ Installing collected packages pynput 56 Successfully installed pynput-1.7.6 57 hotkey keyboard. HotKey( (OCR2) C: |Userslhorva>"S 58 {keyboard.Key.ctrl, keyboard.KeyCode. from_char( 's' ) , keyboard.KeyCode. from char is not recognized as an internal or external command operable program or batch file 59 on_activate) (OCR2) C:|Userslhorvazpip install openai 60 Requirement already satisfied: openai in c:luserslhorvalanaconda3lenvs ocr2|liblsite-packages (1.3.6) 61 with keyboard.Listener( Requirement already satisfied anyio<4,>-3.5.0 in luserslhorvalanaconda3lenvs ocr2|liblsite-packages (from openai) (3.5.0) Requirement already satisfied  distro<2,>=1.7.0 in c:luserslhorva lanaconda3lenvslocr2|liblsite-packages (from openai) (1.8.0) 62 on_press-for_canonical(hotkey.press) Requirement already satisfied httpx<l,>-0.23 0 in c:luserslhorva anaconda3 envs ocr?iliblsite-packages (from openai) (0.23.0) 63 on_release-for_canonical(hotkey.release) ) as listener: Requirement already satisfied pydantic<3,>-1.9.0 in c:luserslhorva anaconda3lenvslocr2|liblsite-packages (from openai) (1.10.12) Requirement already satisfied sniffio in c:luserslhorvalanaconda3 envs ocr2iliblsite packages (from openai) (1.2.0) 64 listener. join() Requirement already satisfied tqdmz4 in c:luserslhorva anaconda3 envs ocr?iliblsite-packages (from openai) (4.65.0) Requirement already satisfied= typing-extensions<5 >=4.5 in c:luserslhorva anaconda3lenvs ocr2|liblsite packages (from openai) (4.7.1 65 66 Requirement already satisfied: idna>-2 8 in c:luserslhorvalanaconda3lenvs ocr2|liblsite-packages (from anyio<4_ >=3 5 0->openai) (3.4) Requirement already satisfied certifi in c:luserslhorva anaconda3 envs ocr2|liblsite packages (from httpx<l,>-0.23.0->openai) (2023 67 11.17) Requirement already satisfied: rfc3986<2,>=1.3 in c:luserslhorvalanaconda3 envs ocr2|liblsite-packages (from rfc3986[idna2008]<2,>-1_ 3->httpx<l >=0.23 0->openai) (1.4.0) Requirement already satisfied: httpcore<0.16.0,>-0.15.0 in c:luserslhorva anaconda3lenvslocr2|liblsite-packages (from httpx<l,>-0.23_ 0->openai) (0.15 0) Requirement already satisfied: colorama in c:luserslhorvalanaconda3lenvs locr2 liblsite-packages (from tqdm-4->openai) (0.4.6) Requirement already satisfied hll<0.13,>=0.11 in c:luserslhorvalanaconda3' envs ocr?iliblsite-packages (from httpcore<0.16.0,>-0.15.0 ~>httpx<l,>=0.23 0->openai) (0.12.0) for_canonicalo lambda (k) Run: main X (OCR2) C: |Userslhorva> C: |UsersIhorvalanaconda3|envs | OCR2Ipython.exe €: |Userslhorva|Documents|Private|Coding|Projects| ScreenGPTImain 1 1 P Git Run ETODO Problems 2 Terminal Python Packages Python Console Services Packages installed successfully: Installed packages: 'OpenAI' (3 minutes ago) 55.46 CRLF UTF-8 spaces Python 3.11 (OCRZ) (2) P main 9PC ENG 19.08 Search c) 8 Stark bewolkt INTL 04/01/2024"""
# user_message = "how to istall pytorch"
# print(query_custom_gpt(screen_content, user_message))