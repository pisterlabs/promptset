import os
import openai
from cortana.model.text_to_audio import text_to_speech_pyttsx3, text_to_speech_TTS, text_to_speech_gtts
from cortana.model.audio_to_text import speech_to_text, stt_whisper, wait_for_call
from cortana.model.predefined_answers import predefined_answers, text_command_detector
from cortana.model.external_classes import DDG, PythonREPL
import webbrowser
import numpy as np
import pathlib
import os
import json 
import time 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

tools = [
    Tool(
        name = "Python REPL",
        func=PythonREPL().run,
        description="useful for when you need to use python to answer a question. You should input python code"
    ),
    Tool(
        name='wikipedia',
        func= WikipediaAPIWrapper().run,
        description="Useful for when you need to look up a topic, country or person on wikipedia"
    ),
    Tool(
        name='DuckDuckGo Search',
        func= DDG.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)]


basedir_model = os.path.dirname(__file__)

with open(os.path.join(basedir_model, 'parameters.json')) as json_file:
    kwargs = json.load(json_file)
    
class Cortana():
    def __init__(self, model_name=None,  language='english', role='Generic', api_key=None, **kwargs):

        # Initialize 
        self.set_language(language)
        self.set_api_key(api_key)
        self.answers = predefined_answers[language]
        self.log = None
        self.flag = True
        self.messages={}
        self.role = role        

        # Model to use
        if model_name is not None:
            self.set_model(model_name, **kwargs)
        else:
            model_name = 'langchain'
        print('-------------------------------------- ')
        print(f'# Cortana + {model_name} ({language}) ')
        print('-------------------------------------- ')
        print('                 ~*An app by ManuNeuro*')


        # Option of voice
        self.tts_option = None
        self.stt_option = None
        
    
    def select_model_source(self, **kwargs):
        
        if 'gpt' in self.model_name:
            self.llm = ChatOpenAI(model=self.model_name, **kwargs)
        else:
            raise NotImplementedError(f'The model {self.model_name} is not implemented yet.')
        
    def greetings(self):
        try:
            self.voice_cortana(self.answers['text_start'], **kwargs['voice'])   
        except:
            pass

    def set_language(self, language):
                
        # Language selector
        if language == 'english':
            self.language = language
            self.code_language = 'en-US'
        elif language == 'french':
            self.language = language
            self.code_language = 'fr-FR'
        else:
            raise Exception("{language}, language not supported.")
        return self.code_language
            
    @staticmethod
    def list_model():
        res = openai.Model.list()
        list_name = [model['id'] for model in res['data']]
        print(list_name)
    
    @staticmethod        
    def set_api_key(api_key=''):
        if api_key is None or api_key=='':
            from cortana.api.encrypt import decrypt_key
            path_key = basedir_model.split('model')[0] + 'api\\'
            api_key = decrypt_key(path_key, 'encrypted_key')
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key

    
    def set_role(self, role, print_):
        self.role = role
        if print_:
            print(f'*>> I changed my role to {self.role}.*')

    def set_model(self, model_name, memory='all', print_=False, **kwargs):
        self.model_name = model_name        
        self.select_model_source(**kwargs)
        
        if memory == 'summary':
            self.memory = ConversationSummaryBufferMemory(
                            llm=self.llm,
                            max_token_limit=3000,
                            )
            self.conversation = ConversationChain(
                llm=self.llm, memory=self.memory
                )
        elif memory == 'all':
            self.memory = ConversationTokenBufferMemory(llm=self.llm, 
                                                        max_token_limit=8000)
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory
            )
        else:
            raise NotImplementedError(f'{memory}: this memory setting is not implemented')
        
        # tools.append(
        #     Tool(name='Prompt question',
        #     func= self.conversation,
        #     description="To answer a regular question which doesn't necessitate internet of specific tools"
        #         )
        #     )
        
        if print_:
            print(f'*>> I changed my model to {self.model_name}.*')

    def prompt(self, input_):
        # Parameters
        if not isinstance(input_, str):
            raise Exception(f'input must be of type str, currently is {type(input_)}')
        else:
            self.last_input = input_
        
        answer = self.conversation(input_)
        
        # Update internal history
        self.messages = self.conversation.memory.buffer
        self.last_answer = answer['response']
        
        print(f'Cortana: {self.last_answer}')
        print('\n----------- \n')

    def prompt_image(self, input_, n=5, size="1024x1024", **kwargs):
                
        if not isinstance(input_, str):
            raise Exception(f'input must be of type str, currently is {type(input_)}')
        else:
            self.last_input = input_
        
        # Update message list
        response = openai.Image.create(prompt=self.last_input, n=n, size=size, **kwargs)
        urls = [response['data'][i]['url'] for i in range(n)]
        [webbrowser.open(url, new=0, autoraise=True) for url in urls]  # Go to example.com
        print('----------------------')
        print(f'Prompt for the image: {self.last_input} ')
        print('----------------------')
        print('I generated image(s) at the following url(s):')
        for url in urls:
            print(str(url)+'')
        print('\n----------- ')
        print('$~~~~~~~~~~~$')
        # self.messages.append({'role':'assistant', "content":urls})
        # self.image_url = url
        return urls
        
    def voice_cortana(self, text, **kwargs):
        
        self.tts_option = kwargs['tts'].get('option', 'gtts')
        
        if self.language == 'french':
            if self.tts_option == "pyttsx3": # Can't use pyttsx3 for french, default is gtts
                tts_option = "gtts"
        
        if self.tts_option=="pyttsx3":
            tts_model = kwargs.get('model', 'Zira')
            text_to_speech_pyttsx3(self, text, tts_model)
        elif self.tts_option=="gtts":
            text_to_speech_gtts(self, text, language=self.code_language)
        elif self.tts_option=='tts':
            tts_model = kwargs['tts'].get('model', None)
            text_to_speech_TTS(self, text, language=self.code_language, model=tts_model)
        else:
            raise Exception(f'Text-to-speech {self.tts_option} option is not supported.')
    
    def listen_cortana(self, *args, **kwargs):
        print(self.answers['stt']['text_sending'])
        self.submit_prompt(*args, _voice=True, **kwargs)
        self.voice_cortana(self.last_answer, **kwargs['voice'])
    
    def cortana_listen(self, stt_option='google', nbtrial=2, timeout=4, **kwargs):
        
        self.stt_option = stt_option
        
        condition=True
        error=None
        text = None
        counter=0
        while condition:
            
            if self.flag:
                if stt_option == 'sphinx' or stt_option == 'google': # Only working in english
                    # This option is to be privileged, it is faster, more reliable, and the output is cleaner
                    response = speech_to_text(self, stt_option, self.code_language, timeout)
                elif stt_option == 'whisper':
                    # Please, note this option is not working well and would require further fine tuning
                    model = kwargs.get('model', 'base')
                    response = stt_whisper(model, timeout)
                else:
                    raise Exception(f'Model {stt_option} of speech-to-text recognition not implemented.')
                
                success = response['success']
                text = response['transcription']
                error = response['error']
                
                if success and text is not None:
                    condition = False
                    
                print('response', response)
                    
                if response['error'] is not None:
                    if "No voice detected." != response['error']:
                        nbtrial += 1 # Request not understood, add one more trial
            
                counter+=1 # Increment counter
                if counter>=nbtrial:
                    condition = False
            else:
                condition = False
                
        return text, error
    
    def talk_with_cortana(self, **kwargs):
        print('------- ')
        print(self.answers['active_title'])
        print('------- ')

        stt_option = kwargs['voice']['stt'].get('option', 'google')
        stt_timeout = kwargs['voice']['stt'].get('timeout', 3)
        stt_nbtrial = kwargs['voice']['stt'].get('nbtrial', 2) # Number of trial for the speech-to-text
        stt_model = kwargs['voice']['stt'].get('model', 'base')
        
        # Language selector
        if self.language == 'french':
            # For french you can't use pyttsx3 ! Default gtts will be used.
            if kwargs['voice']['stt'].get('option', 'gtts') == 'pyttsx3':
                kwargs['stt_option'] = 'gtts'
        
        # Welcoming message
        self.voice_cortana(self.answers['active_mode'], **kwargs['voice'])
        condition = True
        while condition:
            # Get the text from your audio speech
            text, error = self.cortana_listen(stt_option=stt_option, nbtrial=stt_nbtrial, 
                                                       timeout=stt_timeout, model=stt_model)
            
            # Check if a specific command has been used
            if text is not None:
                command = text_command_detector(text, self.language)
            else:
                command = ''
            
            print('command', command)
            
            # Send the text to cortana
            if self.flag:
                # Put cortana in pause
                if command == 'activated_exit':
                    condition = False
                elif (command == 'activated_pause') or ("No voice detected." == error):
                    self.voice_cortana(self.answers['text_idle'], **kwargs['voice'])
                    condition = wait_for_call(self, self.answers['commands']['idle_start'], self.answers['commands']['idle_exit'], self.code_language)
                    if condition:
                        self.voice_cortana(self.answers['response'], **kwargs['voice'])
                elif (command is None) and (text is not None and error is None):
                    self.listen_cortana(text, **kwargs) 
            else:
                condition = False
        self.voice_cortana(self.answers['text_close'], **kwargs['voice'])
        time.sleep(2)
        print(self.answers['protocol_terminated'])
        self.flag = False
        
        # If spinner is still running, stop it
        if hasattr(self, 'spinner'):
            if self.spinner.running:           
                self.spinner.stop()  
      
    def get_search_agent(self, input_text):

        # create and return the chat agent
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=1024, memory_key="chat_history"),
            max_iterations=10,
            min_iterations=5
        )    
        
        output = agent(input_text)['output']
        
        self.messages = self.conversation.memory.buffer
        self.last_answer = output
        
        print(f'Cortana: {self.last_answer}')
        print('\n----------- \n')
       
    def submit_prompt(self, input_text, _voice=False, 
                      preprompt_path=f'{basedir_model}\\preprompt.json',
                      **kwargs):# ./model/preprompt.json
        
        # Check if a specific command has been used

        with open(preprompt_path) as json_file:
            preprompt = json.load(json_file)
        
        if self.messages == {}: # The first time set the persona
            tool_names = [tool.name for tool in tools]
            tool_names.append('Prompt image')
            model_spec = f'You are based on an LLM using the model: {self.model_name}. You have multiple tools at your disposal, that the user can prompt: {tool_names}.'
            template = f"Your persona: {self.answers['persona']}\n" + model_spec
        else:
            template = ''
        
        # Set the role selected in the app
        template = template + "Your specific role for this question is: {role}\n The mode of answer: {mode}\n Question:{question}"  
        
        # The prompt for the question
        prompt_template = PromptTemplate(
            input_variables=["role", "mode", "question"],
            template=template,
        )

        # Check if a specific command is requested        
        if '*' in self.role: 
            # The command is requested in the role of the bot
            command, text_command = text_command_detector(preprompt['roles'][self.role], self.language)
        else:
            # The command is requested in the text by user
            command, text_command = text_command_detector(input_text, self.language)
            
        # If the command is requested directly in the text, remove it
        if text_command is not None:
            if text_command in input_text:
                input_text = input_text.split(text_command)[1]
        
        # Specific mode of answer when generating output (text or voice)
        if not _voice: 
            massage = prompt_template.format(role=preprompt['roles'][self.role], mode=preprompt['text'], question=input_text)
        else:
            massage = prompt_template.format(role=preprompt['roles'][self.role], mode=preprompt['voice'], question=input_text)
        # Issue: this is suboptimal because it's going to be sent at each request, consuming token for redundant information. 
        # However, this is needed for being able to change role at each questions, and jump from active to passive mode. 
        # ToDo: fine a simpler way to only send in new roles, and not recall the role at each request.
        
        print('-------- ')
        print(f'{os.getlogin()}: {input_text} ')
        print('\n----------- ')
        
        
        if command is not None:
            if 'prompt_image' in command:
                print('Cortana: ', self.answers['prompt_image'])
                self.prompt_image(input_text, **kwargs)
                return True
            elif 'langchain_tools' in command:
                self.get_search_agent(massage)
            else:
                print("I didn't understand your prompt, please reformulate.")
                return True
        else:
            # Regular chat
            self.prompt(massage)     
            return True
        
    def show_history(self):
        history = self.conversation.memory.load_memory_variables(
            inputs=[]
        )['history']

        print(history)
    
    def reset_messages(self):
        self.messages={}
        self.conversation.memory.clear()
        
        
# my_cortana = Cortana(language='french')
# my_cortana.set_model('gpt-4', temperature=0.4, max_tokens=500)
# my_cortana.submit_prompt('Write down the Caushy-Shwartz inequality in n dimensions')
# my_cortana.talk_with_cortana(**kwargs)