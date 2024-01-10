import openai
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')
API_KEY = config.get('GPT', 'api_key')
openai.api_base='https://api.ai.cs.ac.cn/v1'
openai.api_key=API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

llm = OpenAI(
        model='gpt-3.5-turbo',
        openai_api_base='https://api.ai.cs.ac.cn/v1/chat',
        openai_api_key=API_KEY
    )
text = r"""Your goal is to generalizing a scheme of attack techniques from the given input of attack technique descriptions.

All output must be in JSON format and follow the schema specified above. Do not output anything except for the extracted information. Do not add any clarifying information. Do not add any fields that are not in the schema. If the text contains attributes that do not appear in the schema, please ignore them.
Here is the output schema:
```
{"properties": {"Slot": {"items": {"type": "object"}, "title": "Slot", "type": "array"}}, "required": ["Slot"]}
```

Generate just one output without any additional information.

Here are some examples:
Input: The adversary renamed ntdsaudit.exe to msadcs.exe.
Output: {"Slot": [{"Renamed Utilities": ["ntdsaudit.exe"]}, {"Renamed Strings": ["msadcs.exe"]}]}
Input: APT28 has used a variety of public exploits, including CVE 2020-0688 and CVE 2020-17144, to gain execution on vulnerable Microsoft Exchange; they have also conducted SQL injection attacks against external websites.
Output: {"Slot": [{"CVE-ID": ["CVE 2020-0688"]}, {"Exploited Vulnerablility Type": ["SQL injection attacks"]}, {"Vulnerable Programs": ["Microsoft Exchange"]}]}
Input: An APT3 downloader creates persistence by creating the following scheduled task: schtasks /create /tn "mysc" /tr C:\Users\Public\test.exe /sc ONLOGON /ru "System".
Output: {"Slot": [{"Task Name": ["mysc"]}, {"Task Run": ["C:\Users\Public\test.exe"]}, {"Schedule Type": ["User"]}]}
Input: STARWHALE has the ability to create the following Windows service to establish persistence on an infected host: sc create Windowscarpstss binpath= "cmd.exe /c cscript.exe c:\windows\system32\w7_1.wsf humpback_whale" start= "auto" obj= "LocalSystem".      
Output: {"Slot": [{"Service Name": ["Windowscarpstss"]}, {"Binary Path": ["cmd.exe /c cscript.exe c:\windows\system32\w7_1.wsf humpback_whale"]}, {"Start Type": ["auto"]}, {"Service Account": ["LocalSystem"]}]}
Input: Dragonfly has used VPNs and Outlook Web Access (OWA) to maintain access to victim networks.
Output: {"Slot": [{"Access Method": ["VPNs", "Outlook Web Access (OWA)"]}]}


Here is real input:
Input: Bumblebee can achieve persistence by copying its DLL to a subdirectory of %APPDATA% and creating a Visual Basic Script that will load the DLL via a scheduled task.
Output: """

# chat_completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[{ "role": "user", "content": text }]
# )
# print(chat_completion.choices[0].message.content)

examples = [
    {
        'example_input': 'The adversary renamed ntdsaudit.exe to msadcs.exe.',
        'example_output': '{"Slot": [{"Renamed Utilities": ["ntdsaudit.exe"]}, {"Renamed Strings": ["msadcs.exe"]}]}'
    },
    {
        'example_input': 'APT28 has used a variety of public exploits, including CVE 2020-0688 and CVE 2020-17144, to gain execution on vulnerable Microsoft Exchange; they have also conducted SQL injection attacks against external websites.',
        'example_output': '{"Slot": [{"CVE-ID": ["CVE 2020-0688"]}, {"Exploited Vulnerablility Type": ["SQL injection attacks"]}, {"Vulnerable Programs": ["Microsoft Exchange"]}]}'
    },
    {
        'example_input': 'An APT3 downloader creates persistence by creating the following scheduled task: schtasks /create /tn "mysc" /tr C:\\Users\Public\\test.exe /sc ONLOGON /ru "System".',
        'example_output':'{"Slot": [{"Task Name": ["mysc"]}, {"Task Run": ["C:\\Users\\Public\\test.exe"]}, {"Schedule Type": ["User"]}]}'
    },
    {
        'example_input': 'STARWHALE has the ability to create the following Windows service to establish persistence on an infected host: sc create Windowscarpstss binpath= "cmd.exe /c cscript.exe c:\\windows\\system32\\w7_1.wsf humpback_whale" start= "auto" obj= "LocalSystem".',
        'example_output': '{"Slot": [{"Service Name": ["Windowscarpstss"]}, {"Binary Path": ["cmd.exe /c cscript.exe c:\\windows\\system32\\w7_1.wsf humpback_whale"]}, {"Start Type": ["auto"]}, {"Service Account": ["LocalSystem"]}]}'
    },
    {
        'example_input': 'Dragonfly has used VPNs and Outlook Web Access (OWA) to maintain access to victim networks.',
        'example_output': '{"Slot": [{"Access Method": ["VPNs", "Outlook Web Access (OWA)"]}]}'
    }
]

prompt_template = CandicatedTTPSchemaGeneratePromptTemplate(examples)
print()