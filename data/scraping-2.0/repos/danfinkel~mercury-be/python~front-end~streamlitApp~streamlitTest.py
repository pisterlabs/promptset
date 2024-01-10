from openai import chat
import streamlit as st
from code_editor import code_editor
import asyncio
import json
import requests as re

import aiohttp

STICKY_HEADER = """
        <style>
            div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                position: sticky;
                top: 2.875rem;
                background-color: #0f1117;
                z-index: 999;
            }
            .fixed-header {
                border-bottom: 1px solid black;
            }
        </style>
            """

CHAT_ICONS = {
    "dan": "⛹️",
    "Engineer": "⛹️",
    "Student": "👩‍🎓",
    "databaseAdmin": "🖥️",
    "user": "🦁",
    "sys_admin": "🔧",
    "robot": "🤖"
}

# URL = 'https://mercury-jzz5.onrender.com/promptAI'
PROMPT_URL = "http://127.0.0.1:5000/promptAI"
TEACHING_URL = "http://127.0.0.1:5000/teachAI"

prompts = {'reach': 'How many users saw an ad?',
           '7-Day Daily Reach': 'Please report daily campaign reach where reach for a given day is defined to be total number of users who were exposed in the previous 7 day window. Perform the calculation for each day from August 1 2023 to September 1 2023',
           'impressions': 'How many ads were served?',
           'lift': 'What was the lift of the campaign?',
           '7-Day Daily Lift': 'Please report daily campaign lift where lift for a given day is defined to be the conversion rate of users exposed divided by the conversion rate of users unexposed to the ad campaign. Conversion rates utilize causality in that a user is considered converted only if they convert within 7 days of the date under evaluation. Perform the calculation for each day from August 1 2023 to September 1 2023.'
           }

LEARNING_PROMPT = """This is very helpful thank you! 
I would like to format your recommendation so that I can store
it to enhance future prompts to AI. Can you respond to this prompt
by providing

1. A short, concise and detailed summary of your instructions. This summary
should be approximately ten words and will be used to determine if your instructions
should be added to a future AI prompt. Please precede your summary with the text "SUMMARY:" 
so I can easily find it.

2. Your best set of instructions. Please precede your instructions with the 
text "INSTRUCTIONS:" so I can easily find them.

Thank you!
"""


class StreamlitPage():
    def __init__(self, page_title):
        self.page_title = page_title        

    @property
    def page_title(self):
        return self._page_title
    
    @page_title.setter
    def page_title(self, value):
        self._page_title = value

class StreamlitChatPage(StreamlitPage):
    def __init__(self, page_title):
        super().__init__(page_title) # type: ignore        

        st.set_page_config(page_title=self.page_title)        
        self.tabs = st.tabs(["AI Chat", 
                             "Summary", 
                             "Run Python Code", 
                             "Audit Results", 
                             "Save for Reuse",
                             "Teach Your Assistant"]
                            )
        self.ai_question = 'No Question Yet Submitted'
        self.ai_code = 'No Python Yet Generated'
        self.ai_answer = "No AI Answer Yet Generated"

        self.ai_response = {
                            "question": 'No Question Yet Submitted',
                            "code": 'No Code Yet Generated',
                            "answer": 'No AI Answer Has Been Created'
                            }
        
    @property
    def ai_response(self):
        return self._ai_response
    
    @ai_response.setter
    def ai_response(self, value):
        self._ai_response = value
        if 'chat_responses' in st.session_state.keys() and len(st.session_state.chat_responses) > 0:
            for chat in st.session_state.chat_responses:
                if 'QUESTION' in chat["content"]:
                    self._ai_response["question"] = chat["content"].split('QUESTION:')[1].split('PYTHON SCRIPT')[0]
                if 'PYTHON SCRIPT' in chat["content"]:
                    self._ai_response["code"] = chat["content"].split('PYTHON SCRIPT:')[1].split('ANSWER')[0]
                if 'ANSWER' in chat["content"]:
                    self._ai_response["answer"] = chat["content"].split('ANSWER:')[1]
                    break

    def _top_teaching_page(self):    
        header = st.container()
        header.title('📈💬 Mercury AI - Classroom 📚', help="AI Enabled Agents Prompted to Solve Data Science Tasks. This application is pointed at a Postgres database with a set of (fake) exposures for an advertising campaign for Bob's Hamburgers. There is also a (fake) conversions file and a (fake) universe file.")

        teaching_form = header.form("my_teaching_form")
        self.promptForTeaching = teaching_form.text_area("Hey 🦁 Talk to Roger", key="teaching_input", height=150)
        col1, col2 = teaching_form.columns([.15,1])
        self.runTeaching = col1.form_submit_button("Submit")
        self.storeLesson = col2.form_submit_button("Summarize and Store")

        header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

        ### Custom CSS for the sticky header
        st.markdown(
            STICKY_HEADER,
            unsafe_allow_html=True
        )    

    def _top_ai_page(self):    
        header = st.container()
        header.title('📈💬 Mercury AI - Problem Solving', help="AI Enabled Agents Prompted to Solve Data Science Tasks. This application is pointed at a Postgres database with a set of (fake) exposures for an advertising campaign for Bob's Hamburgers. There is also a (fake) conversions file and a (fake) universe file.")

        learningform = header.form("my_form")
        self.promptForAI = learningform.text_area("Hey 🦁 enter your request here", key="text")
        colA, colB = learningform.columns([.15,1])
        self.runAI = colA.form_submit_button("Submit")
        self.useRoger = colB.toggle("Use Memory-Enhanced AI", value=False, key="useRoger")        

        header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

        ### Custom CSS for the sticky header
        st.markdown(
            STICKY_HEADER,
            unsafe_allow_html=True
        )            
        
        # st.write("AI Enabled Agents Prompted to Solve Data Science Tasks. This application is pointed at a Postgres database with a set of (fake) exposures for an advertising campaign for Bob's Hamburgers. There is also a (fake) conversions file and a (fake) universe file.")        
        # st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    def _set_chat(self):
        if 'chat_responses' not in st.session_state:            
            st.session_state.chat_responses = [{"chatName": "Engineer", "content": "Hello, I am your AI Assistant powered by Mecury Analytics"}]
            st.session_state.chat_responses.append({"chatName": "sys_admin", "content": "And I am the System Admin that will help you work with your AI Assistant. How can we help you?"})

    def _set_teachchat(self):
        if 'teachchat_responses' not in st.session_state:            
            st.session_state.teachchat_responses = [{"chatName": "Student", "content": "Hello, I am Roger. I am an AI Assistant that is here to learn"}]
            st.session_state.teachchat_responses.append({"chatName": "Student", "content": "What topic should we learn about today? My goal is to work with you to create a summary of an adtech topic which can be used to enrich future prompts and improve overall AI performance."})

    def _populate_teachchat(self):
        self._set_teachchat()
        for msg in st.session_state.teachchat_responses:
            with st.chat_message(name=msg["chatName"], avatar=CHAT_ICONS.get(msg["chatName"])):
                st.write(str(msg["content"]))

    def _populate_chat(self):
        self._set_chat()
        for msg in st.session_state.chat_responses:
            with st.chat_message(name=msg["chatName"], avatar=CHAT_ICONS.get(msg["chatName"])):
                st.write(str(msg["content"]))                

    def _build_menu(self):
        with st.sidebar:
            st.title('⚙️ Configuration')

            self.api_secret = st.text_input("Enter API Secret")

            self.selected_prompt = st.selectbox(label="Pre-Build Prompts", 
                                                options=list(prompts.keys()), 
                                                index=0, 
                                                )
            self.menu_preview = st.button("Preview Prompt", on_click=self.on_preview_click)
            
            self.clear_button = st.button("Clear", on_click=self.clear_chat)  
    
    def on_preview_click(self):
        st.session_state.prompt = prompts[self.selected_prompt] # type: ignore      
        st.session_state.text = prompts[self.selected_prompt] # type: ignore

        # self._populate_chat()

    def clear_chat(self):
        st.session_state.pop("chat_responses")
        st.session_state.text = None


    async def chatWAI(self,
                      promptForAI: str,  # type: ignore
                      responses: list[dict],
                      url: str,
                      **kwargs):
        """
        Inspired by: https://stackoverflow.com/questions/59681726/how-to-read-lines-of-a-streaming-api-with-aiohttp-in-python
        and: https://stackoverflow.com/questions/74800726/running-asyncio-with-streamlit-dashboard
        General async Tutorial here: https://realpython.com/python-async-features/
        """
        with st.chat_message(name='user', avatar=CHAT_ICONS.get('user')):
            st.write(promptForAI)
            responses.append({"chatName": 'user', "content": promptForAI})    

        async with aiohttp.ClientSession() as session:
            print(f"Active thread is: {st.session_state.get('active_thread_id', '')}")

            data_dict = {
                    "prompt": promptForAI, 
                    "thread_id": st.session_state.get('active_thread_id', '')
                    }
            data_dict.update(kwargs)

            async with session.post(url, data=data_dict) as r: # type: ignore
                async for line in r.content:
                    formatted_line = line.replace(b'"', b'\\\'').decode('utf-8').replace("'",'"')                    
                    try:
                        # try to extract content and user from the json
                        content = json.loads(formatted_line).get("content")
                        chatName = json.loads(formatted_line).get("user")

                    except json.decoder.JSONDecodeError:

                        # json decoder didnt work so fish out the
                        # content and chat name
                        print('ERROR ERROR ERROR')
                        print(line)

                        formatted_line = line.decode('utf-8')
                        content = formatted_line.split("""content""")[1][4:][:-3].replace('\\n','\n').replace('\\\\','\\')
                        chatName = formatted_line.split('user')[1].split(',')[0][4:][:-1]

                    if chatName != 'sys_internal':
                        with st.chat_message(name=chatName, avatar=CHAT_ICONS.get(chatName)):
                            st.write(str(content))
                            responses.append({"chatName": chatName, "content": str(content)})
                    else:
                        print(f'here for active_thread_id {json.loads(formatted_line)["thread_id"]}')
                        st.session_state.active_thread_id = json.loads(formatted_line)["thread_id"]
                        print(f"just set active thread id to be {st.session_state.active_thread_id}")

    def initialize(self):
        with self.tabs[0]:
            self._top_ai_page()        
            self._build_menu()
            
            self._populate_chat()  

            # self._chat_input()      

            if self.runAI:
                if self.promptForAI != '':
                    st.session_state.prompt = self.promptForAI # type: ignore
                    print(f"self.useRoger is: {st.session_state.get('useRoger', False)}")
                    asyncio.run(self.chatWAI(st.session_state.prompt, 
                                             responses=st.session_state.chat_responses,
                                             url=PROMPT_URL,
                                             useTeachableAI = st.session_state.get("useRoger", False)))
                else:
                    st.warning('Enter a Prompt!') # type: ignore
        
        with self.tabs[1]:
            self.ai_response = {
                            "question": 'No Question Yet Submitted',
                            "code": 'No Code Yet Generated',
                            "answer": 'No AI Answer Has Been Created'
                            }
            st.markdown('## Analytic Question:\n\n' + self.ai_response["question"])
            st.markdown('## Python Script:\n\n' + self.ai_response["code"])
            st.markdown('## Answer:\n\n' + self.ai_response["answer"])
        
        with self.tabs[2]:
            run_python_form = st.form("Run Python")
            with run_python_form:
                st.markdown("## Python Script")

                # with open('/Users/danfinkel/github/mercury-be/python/front-end/streamlitApp/buttons.json') as json_button_file_alt:
                #     btns = json.load(json_button_file_alt)                

                btns = [{
                "name": "Copy",
                "feather": "Copy",
                "alwaysOn": True,
                "commands": ["copyAll"],
                "style": {"top": "0.46rem", "right": "0.4rem"}
                },
                {
                    "name": "Run",
                    "feather": "Play",
                    "primary": True,
                    "hasText": True,
                    "showWithIcon": True,
                    "commands": ["submit"],
                    "alwaysOn": True,
                    "style": {"bottom": "0.44rem", "right": "0.4rem"}
                },                
                ]
                    
                response_dict = code_editor(self.ai_response["code"], 
                                            lang="python", 
                                            key="code_editor_demo", 
                                            height = [19, 22],  # type: ignore
                                            props = {"style": {"borderRadius": "0px 0px 8px 8px"}},
                                            buttons=btns)
                # if len(response_dict) > 0:
                #     if "submit" in [r["type"] for r in response_dict]:
                #         st.write('Run Pressed!')
                run_python_button = st.form_submit_button("Run Python")
                if run_python_button:
                    print('here')
                    st.write(response_dict)
                    st.write(st.session_state.get("code_editor_demo", "no code"))
                    # response = re.post('http://127.0.0.1:5000/runPython', json={'pythonScript': self.ai_response["code"].replace('```python', '').replace('```', '').replace('```python', '')})
                    # response = re.post('http://127.0.0.1:5000/runPython', json={'pythonScript': self.ai_response["code"].replace('```python', '').replace('```', '').replace('```python', '')})
                    print('here2')
                    # st.write(response.text)
        
        with self.tabs[5]:
            self._top_teaching_page()
            self._populate_teachchat()

            if self.runTeaching:
                if self.promptForTeaching != '':
                    st.session_state.teachingPrompt = self.promptForTeaching # type: ignore                    
                    self.promptForTeaching = ''

                    print(f"prompt is: {st.session_state.teachingPrompt}")                    
                    asyncio.run(self.chatWAI(st.session_state.teachingPrompt, 
                                             responses=st.session_state.teachchat_responses,
                                             url=TEACHING_URL))

                else:
                    st.warning('Say Something!') # type: ignore

            if self.storeLesson:
                print(len(st.session_state.teachchat_responses))
                if st.session_state.get('active_thread_id', '') == '':
                    st.warning('You have to chat before you can store a lesson!') # type: ignore
                else:
                    asyncio.run(self.chatWAI(LEARNING_PROMPT, 
                                             responses=st.session_state.teachchat_responses,
                                             url=TEACHING_URL,
                                             save_result=True))


cp = StreamlitChatPage(page_title='📈💬 Mercury AI')
cp.initialize()  