import os
import sys
import time 
import datetime

from langchain.chains import LLMChain
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
# Obtain the parent directory
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from chatbot.prompt_config import PromptConfig_v1
from chatbot.conversation_manager import ConversationManager_v1
from chatbot.conversation_analysis import ConversationAnalysis_v1, ConversationAnalysis_v2, ConversationAnalysis_v3
from chatbot.chatbot import Chatbot_v1
from chatbot.speech_to_text import S2T_with_whisper, S2T_with_openai
from chatbot.text_to_speech import T2S_with_openai

from users_manager.user_manager import UserManager

from configurations.config_llm import FT_NAME, CHAT_TEMPERATURE, OPENAI_API_KEY
from configurations.config_prompts import SUMMARY_CONVERSATION_v1, PERSONAL_INFORMATION_v1, EVE_PROMPT_v1, SEARCH_IN_VECTORSTORE_DATABASE_v1
from configurations.config_templates import PERSONAL_INFORMATION_TEMPLATE_PATH
from configurations.config_vectordb import VECTOR_DB_PATH

from functions_auxiliary import read_txt_files

# Import the vector database searcher

from vector_database.vector_database import VectorDatabase

class BaseEve():
    def __init__(self, llm = None):
        self.model = llm
        pass
    
    def response(self, message): 
        pass
    
class Eve_v1(BaseEve):
    
    def __init__(self,):
        # Configure the API key
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # Initialize the prompt manager
        self.prompt_manager = PromptConfig_v1()
        
        # Initialie the conversation manager
        self.conversation_manager = ConversationManager_v1()
        
        # Initialize the conversation analysis
        self.conversation_analysis = ConversationAnalysis_v1()
        
        # Initialize the chatbot
        self.chatbot = Chatbot_v1()
        
        pass
    
    def initialize(self,):
        # Load prompts
        self.prompt_manager.load_prompts(paths = [SUMMARY_CONVERSATION_v1['path'], PERSONAL_INFORMATION_v1['path'], EVE_PROMPT_v1['path']],
                                         keys = ['summary_conversation', 'personal_information', 'eve_message'],
                                         input_variables = [SUMMARY_CONVERSATION_v1['input_variables'], PERSONAL_INFORMATION_v1['input_variables'], EVE_PROMPT_v1['input_variables']])
        
        # Aca tenemos que chequear si existe o no archivos de la conversacion anteriores 
        
        # Load template of personal information
        self.conversation_analysis.load_personal_information(personal_information = read_txt_files(PERSONAL_INFORMATION_TEMPLATE_PATH))
        # Load prompt for the personal information
        self.conversation_analysis.load_prompt_personal_information(prompt = self.prompt_manager.get_prompt('personal_information'))
        
        # Load prompt for the summary conversation
        self.conversation_analysis.load_prompt_summary_conversation(prompt = self.prompt_manager.get_prompt('summary_conversation'))
        # Load prompt for the chatbot
        self.chatbot.load_prompt(prompt = self.prompt_manager.get_prompt('eve_message'))
             
    def response(self, message):
        # We obtain the last messages
        last_messages = self.conversation_manager.get_n_last_messages()
        
        # Add patient message to memory
        self.conversation_manager.add_message(message, is_ai_message = False)
    
        # Obtain the complete current conversation
        current_conversation = self.conversation_manager.get_conversation()
        
        # Process the personal information
        self.conversation_analysis.update_personal_information(current_conversation)
        # Obtain the new personal information
        personal_info = self.conversation_analysis.get_personal_information()
        
        # Process the summary conversation
        self.conversation_analysis.update_summary_conversation(last_messages , current_conversation)
        # Obtain the new summary conversation
        summary_conversation = self.conversation_analysis.get_summary_conversation()
        
        # Obtain the response of the chatbot
        eve_message = self.chatbot.response(message = message,
                                            personal_information = personal_info,
                                            previous_conversation_summary = summary_conversation,
                                            last_messages = last_messages)
        
        # Add AI assistant message to memory
        self.conversation_manager.add_message(eve_message, is_ai_message = True)
        
        return eve_message
    

class Eve_v2(Eve_v1):
    
    def __init__(self,):
        super().__init__()
        pass
    
    def response(self, message):
        # We obtain the last messages
        last_messages = self.conversation_manager.get_n_last_messages()
        
        # Add patient message to memory
        self.conversation_manager.add_message(message, is_ai_message = False)
    
        # Obtain the complete current conversation
        current_conversation = self.conversation_manager.get_conversation()
        
        # Init the threads for the personal information and the summary conversation
        personal_info_thread = threading.Thread(target=self._process_personal_information, args=(current_conversation,))
        summary_thread = threading.Thread(target=self._process_summary_conversation, args=(last_messages , current_conversation))
        personal_info_thread.start()
        summary_thread.start()
        personal_info_thread.join()
        summary_thread.join()
        
        # Obtain the new personal information
        personal_info = self.conversation_analysis.get_personal_information()
        # Obtain the new summary conversation
        summary_conversation = self.conversation_analysis.get_summary_conversation()
        
        # Obtain the response of the chatbot
        eve_message = self.chatbot.response(message = message,
                                            personal_information = personal_info,
                                            previous_conversation_summary = summary_conversation,
                                            last_messages = last_messages)
        
        # Add AI assistant message to memory
        self.conversation_manager.add_message(eve_message, is_ai_message = True)
        
        return eve_message
            
    def _process_personal_information(self,current_conversation):
        self.conversation_analysis.update_personal_information(current_conversation)

    def _process_summary_conversation(self,last_messages , current_conversation):
        self.conversation_analysis.update_summary_conversation(last_messages , current_conversation)
    

# These Eve versions save the current conversation and the personal information in the user folder
class Eve_v3(Eve_v2):
    def __init__(self,):
        super().__init__()
        
        # User Manager
        self.user_manager = UserManager()
        pass
    
    def initialize(self,):
        # Login process
        self.user_manager.user_login()
        
        # Load prompts
        self.prompt_manager.load_prompts(paths = [SUMMARY_CONVERSATION_v1['path'], PERSONAL_INFORMATION_v1['path'], EVE_PROMPT_v1['path']],
                                         keys = ['summary_conversation', 'personal_information', 'eve_message'],
                                         input_variables = [SUMMARY_CONVERSATION_v1['input_variables'], PERSONAL_INFORMATION_v1['input_variables'], EVE_PROMPT_v1['input_variables']])
        
        # Load the information requiered for the conversation
        self.load_personal_information()
        
        # Load prompt for the summary conversation
        self.load_summary_conversation()
        
        # Load prompt for the chatbot
        self.chatbot.load_prompt(prompt = self.prompt_manager.get_prompt('eve_message'))
    
    def load_personal_information(self,):
        # Obtain the personal information from usermanager
        personal_information = self.user_manager.get_user_information(key='personal_information')
        #Load the personal information
        self.conversation_analysis.load_personal_information(personal_information = personal_information)
        # Load prompt for the personal information
        self.conversation_analysis.load_prompt_personal_information(prompt = self.prompt_manager.get_prompt('personal_information'))
        
    def load_summary_conversation(self,):
        # Load prompt for the summary conversation
        self.conversation_analysis.load_prompt_summary_conversation(prompt = self.prompt_manager.get_prompt('summary_conversation'))
        
    def finalize(self,):
        # Message of process information and wait a minute
        print("Processing the conversation...")
        # Update the personal information
        self.user_manager.update_user_information(key='personal_information', value=self.conversation_analysis.get_personal_information())
        
        # Save the conversation
        self.save_conversation()
        
        # Save the personal information 
        self.save_personal_information()
        
    def save_conversation(self,):
        # Obtain the current conversation
        current_conversation = self.conversation_manager.get_conversation()
        # Obtain the number of the last session
        number_session = self.user_manager.get_user_information(key='last_session_number')
        if number_session is None:
            number_session = 0
        number_session+=1
        # The file name is the current date and time (YYYY/MM/DD)
        date = datetime.datetime.now().strftime("%Y_%m_%d")
        filename = f"{number_session}_session_{date}.txt"
        self.user_manager.save_content(content = current_conversation, 
                                       folder = 'sessions', 
                                       filename = filename)
        
    def save_personal_information(self,):
        # Save the personal information
        self.user_manager.save_user_information()


class Eve_v4(Eve_v3):
    def __init__(self,):
        super().__init__()
        self.audio_transcriber = S2T_with_whisper()
        
    def response_with_audio(self, input_user, audio_flag=False):
        # Check if the input is a text or a audio
        if audio_flag:
            # If the input is a audio, we process the audio
            input_user = self.audio_transcriber.transcribe(input_user)
        # Obtain the response of Eve
        eve_message = self.response(input_user)
        return eve_message
    
class Eve_v4a(Eve_v3):
    def __init__(self,):
        super().__init__()
        self.audio_transcriber = S2T_with_openai()
        
    def response_with_audio(self, input_user, audio_flag=False):
        # Check if the input is a text or a audio
        if audio_flag:
            # If the input is a audio, we process the audio
            input_user = self.audio_transcriber.transcribe(input_user)
        # Obtain the response of Eve
        eve_message = self.response(input_user)
        return eve_message
    
class Eve_v4b(Eve_v3):
    def __init__(self,):
        super().__init__()
        
        "Esta version ya tiene incorporada la parte de tts stt."
        self.audio_transcriber = S2T_with_openai()
        self.audio_generator = T2S_with_openai()
        
    def response_with_audio(self, input_user, audio_flag=False, audio_response_flag = False):
        # Check if the input is a text or a audio
        if audio_flag:
            # If the input is a audio, we process the audio
            input_user = self.audio_transcriber.transcribe(input_user)
        # Obtain the response of Eve
        eve_message = self.response(input_user)
        
        if audio_response_flag:
            # If the response is a audio, we process the audio
            audio_path_file = self.audio_generator.create(eve_message)
            return eve_message, audio_path_file
        else:
            return eve_message
    
class Eve_v5(Eve_v4b):
    def __init__(self,):
        super().__init__()
        """ Aca incorpora la busqueda en la base de datos vectorial"""
        
        # Initialize the conversation analysis
        self.conversation_analysis = ConversationAnalysis_v2()
        
        self.vectordabase_serch = VectorDatabase(db_path=VECTOR_DB_PATH)
        
    def initialize(self,):
        # Load prompts
        self.prompt_manager.load_prompts(paths = [SUMMARY_CONVERSATION_v1['path'], PERSONAL_INFORMATION_v1['path'], EVE_PROMPT_v1['path'], SEARCH_IN_VECTORSTORE_DATABASE_v1['path']],
                                         keys = ['summary_conversation', 'personal_information', 'eve_message','search_in_database'],
                                         input_variables = [SUMMARY_CONVERSATION_v1['input_variables'], PERSONAL_INFORMATION_v1['input_variables'], EVE_PROMPT_v1['input_variables'], SEARCH_IN_VECTORSTORE_DATABASE_v1['input_variables']])
        
        # Load the information requiered for the conversation
        self.load_personal_information()
        
        # Load prompt for the summary conversation
        self.load_summary_conversation()
        
        # Load the description of the database
        self.load_description_database()
        
        # Load prompt for the chatbot
        self.chatbot.load_prompt(prompt = self.prompt_manager.get_prompt('eve_message'))
        
    def load_description_database(self,):
        description = "Petroleum Geo-Services (PGS) is a leading provider of seismic and reservoir services for the oil and gas industry. The company offers a wide range of geophysical services, including seismic data acquisition, processing, and interpretation. PGS operates a fleet of seismic vessels and has a global presence, with offices and operations in various countries around the world. The company's services are used by oil and gas companies to identify and evaluate potential hydrocarbon reserves, optimize field development plans, and monitor reservoir performance. PGS is committed to delivering high-quality data and insights to its clients, helping them make informed decisions and maximize the value of their assets. "
        self.conversation_analysis.load_description_search_database(description = description)
        # Load prompt for the search database
        self.conversation_analysis.load_prompt_search_database(prompt = self.prompt_manager.get_prompt('search_in_database'))
        
    def response(self, message):
        ## Aca hay que chequear si se entra o no en la busqueda de la base de datos. 
        search_in_database = self.conversation_analysis.search_database(message)
        
        if search_in_database:
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)
            
            # Search in database
            eve_message = self.vectordabase_serch.predict(message)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)
        else:
            # We obtain the last messages
            last_messages = self.conversation_manager.get_n_last_messages()
            
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)

            # Obtain the complete current conversation
            current_conversation = self.conversation_manager.get_conversation()
            
            # Init the threads for the personal information and the summary conversation
            personal_info_thread = threading.Thread(target=self._process_personal_information, args=(current_conversation,))
            summary_thread = threading.Thread(target=self._process_summary_conversation, args=(last_messages , current_conversation))
            personal_info_thread.start()
            summary_thread.start()
            personal_info_thread.join()
            summary_thread.join()
            
            # Obtain the new personal information
            personal_info = self.conversation_analysis.get_personal_information()
            # Obtain the new summary conversation
            summary_conversation = self.conversation_analysis.get_summary_conversation()
            
            # Obtain the response of the chatbot
            eve_message = self.chatbot.response(message = message,
                                                personal_information = personal_info,
                                                previous_conversation_summary = summary_conversation,
                                                last_messages = last_messages)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)

        return eve_message
        
class Eve_v5a(Eve_v5):
    def __init__(self,):
        super().__init__()
        """ Aca incorpora la busqueda en la base de datos vectorial"""
        
        # Initialize the conversation analysis
        self.conversation_analysis = ConversationAnalysis_v3()
        self.n_questions_surveys = 0
    
    def initialize(self,):
        # Load prompts
        self.prompt_manager.load_prompts(paths = [SUMMARY_CONVERSATION_v1['path'], PERSONAL_INFORMATION_v1['path'], EVE_PROMPT_v1['path'], SEARCH_IN_VECTORSTORE_DATABASE_v1['path']],
                                         keys = ['summary_conversation', 'personal_information', 'eve_message','search_in_database'],
                                         input_variables = [SUMMARY_CONVERSATION_v1['input_variables'], PERSONAL_INFORMATION_v1['input_variables'], EVE_PROMPT_v1['input_variables'], SEARCH_IN_VECTORSTORE_DATABASE_v1['input_variables']])
        
        # Load the information requiered for the conversation
        self.load_personal_information()
        
        # Load prompt for the summary conversation
        self.load_summary_conversation()
        
        # Load the description of the database
        self.load_description_database()
        
        # Load prompt for the chatbot
        self.chatbot.load_prompt(prompt = self.prompt_manager.get_prompt('eve_message'))
        
        # Load the survey
        self.load_surveys()
        
        # Init the survey
        self.init_survey()
    
    def init_survey(self,):
        # Check if there is a survey to answer
        if self.n_questions_surveys > 0:
            # Generate the message of the survey
            message = "EVE: I have a survey for you. Do you want to answer it?(Y/N): "
            # Obtain the response of the user
            response = input(message)
            # CHeck if response is Y or N
            if response == "Y":
                print('EVE: Thanks, Lets start with the survey')        
                # Create a for loop to load the questions
                for i in range(self.n_questions_surveys):
                    # Obtain the question
                    question, answers = self.conversation_analysis.obtain_question(i)
                    complete_answered = question + " (choice: a,b,c,d,...)\n"
                    abcd = ['a','b','c','d','e','f','g','h','i','j']
                    for j in range(len(answers)):
                        complete_answered += f"{abcd[j]}. {answers[j]}\n"
                    print(complete_answered)
                    answer = input("Choice:")
                    # Answer lower case
                    answer = answer.lower()
                    # Now we need to match the answer with the index
                    index_answer = abcd.index(answer)
                    real_answer = answers[index_answer]
                    self.conversation_analysis.respond_question( number_question= i, answer=real_answer)

    def load_surveys(self,):
        user_path = self.user_manager.get_user_folder(self.user_manager.username, self.user_manager.password)
        # Survey path
        survey_path = os.path.join(user_path, 'surveys', 'surveys.csv')
        # Load the surveys
        self.n_questions_surveys = self.conversation_analysis.load_survey(path = survey_path)
           
    def load_description_database(self,):
        description = "Petroleum Geo-Services (PGS) is a leading provider of seismic and reservoir services for the oil and gas industry. The company offers a wide range of geophysical services, including seismic data acquisition, processing, and interpretation. PGS operates a fleet of seismic vessels and has a global presence, with offices and operations in various countries around the world. The company's services are used by oil and gas companies to identify and evaluate potential hydrocarbon reserves, optimize field development plans, and monitor reservoir performance. PGS is committed to delivering high-quality data and insights to its clients, helping them make informed decisions and maximize the value of their assets. "
        self.conversation_analysis.load_description_search_database(description = description)
        # Load prompt for the search database
        self.conversation_analysis.load_prompt_search_database(prompt = self.prompt_manager.get_prompt('search_in_database'))
    
    def response_with_audio(self, input_user, audio_flag=False, audio_response_flag = False):
        # Check if the input is a text or a audio
        if audio_flag:
            # If the input is a audio, we process the audio
            input_user = self.audio_transcriber.transcribe(input_user)
        # Obtain the response of Eve
        eve_message = self.response(input_user)
        
        if audio_response_flag:
            # If the response is a audio, we process the audio
            audio_path_file = self.audio_generator.create(eve_message)
            return eve_message, audio_path_file
        else:
            return eve_message
        
          
    def response(self, message):
        ## Aca hay que chequear si se entra o no en la busqueda de la base de datos. 
        search_in_database = self.conversation_analysis.search_database(message)
        
        if search_in_database:
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)
            
            # Search in database
            eve_message = self.vectordabase_serch.predict(message)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)
        else:
            # We obtain the last messages
            last_messages = self.conversation_manager.get_n_last_messages()
            
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)

            # Obtain the complete current conversation
            current_conversation = self.conversation_manager.get_conversation()
            
            # Init the threads for the personal information and the summary conversation
            personal_info_thread = threading.Thread(target=self._process_personal_information, args=(current_conversation,))
            summary_thread = threading.Thread(target=self._process_summary_conversation, args=(last_messages , current_conversation))
            personal_info_thread.start()
            summary_thread.start()
            personal_info_thread.join()
            summary_thread.join()
            
            # Obtain the new personal information
            personal_info = self.conversation_analysis.get_personal_information()
            # Obtain the new summary conversation
            summary_conversation = self.conversation_analysis.get_summary_conversation()
            
            # Obtain the response of the chatbot
            eve_message = self.chatbot.response(message = message,
                                                personal_information = personal_info,
                                                previous_conversation_summary = summary_conversation,
                                                last_messages = last_messages)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)

        return eve_message
    
class Eve_v5b(Eve_v5):
    def __init__(self,):
        super().__init__()
        """Aca se cambio la forma en la cual se inicializa Eve"""
        # Initialize the conversation analysis
        self.conversation_analysis = ConversationAnalysis_v3()
        
        # Atributes for surveys
        self.n_questions_surveys = 0
        self.n_questions_surveys_answered = 0
        self.do_survey = False
        self.survey_answered = False
        
        self.temp_path = None
        self.username = None
        
    def initialize(self,username = None, password = None):
        # Login process
        self.user_manager.user_login(username = username, password = password)
        self.username = self.user_manager.get_username()
        
        # Load temp path
        self.load_tmp_file()
        
        # Load prompts
        self.prompt_manager.load_prompts(paths = [SUMMARY_CONVERSATION_v1['path'], PERSONAL_INFORMATION_v1['path'], EVE_PROMPT_v1['path'], SEARCH_IN_VECTORSTORE_DATABASE_v1['path']],
                                         keys = ['summary_conversation', 'personal_information', 'eve_message','search_in_database'],
                                         input_variables = [SUMMARY_CONVERSATION_v1['input_variables'], PERSONAL_INFORMATION_v1['input_variables'], EVE_PROMPT_v1['input_variables'], SEARCH_IN_VECTORSTORE_DATABASE_v1['input_variables']])
        
        # Load the information requiered for the conversation
        self.load_personal_information()
        
        # Load prompt for the summary conversation
        self.load_summary_conversation()
        
        # Load the description of the database
        self.load_description_database()
        
        # Load prompt for the chatbot
        self.chatbot.load_prompt(prompt = self.prompt_manager.get_prompt('eve_message'))
        
        # Load the survey
        self.load_surveys()
    
    def load_tmp_file(self,):
        self.temp_path = os.path.join('app','TEMP',self.username)
        # Check if path exists and if not create it
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)    
        # Folders to create
        folders_create = ['audio_responses']
        
        for folder in folders_create:
            # Check if path exists and if not create the folder
            if not os.path.exists(os.path.join(self.temp_path, folder)):
                os.mkdir(os.path.join(self.temp_path, folder))
        
    def ask_to_do_survey(self,):
        # Check if there is a survey to answer
        if self.n_questions_surveys > 0:
            # Generate the message of the survey
            message = "I have a survey for you. Do you want to answer it?(Y/N): "
            return  message
        
    def obtain_response_to_do_survey(self,response):
        response = response.upper()
        if response == "Y":
            self.do_survey = True
            self.question_survey_id = 0
        else:
            self.do_survey = False
            
    def get_survey(self, return_dict = False):
        # Check if there is a survey to answer
        if self.do_survey:
            # Check if question_survey_id < n_questions_surveys
            if self.question_survey_id < self.n_questions_surveys -1:
                if return_dict:
                    # Obtain the question
                    return self.conversation_analysis.obtain_question(self.question_survey_id, return_dict=return_dict)
                else:
                    question, answers = self.conversation_analysis.obtain_question(self.question_survey_id)
                    self.answers = answers
                    complete_question = question + " (choice: a,b,c,d,...)\n"
                    abcd = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',]
                    for j in range(len(answers)):
                        complete_question += f"{abcd[j]}. {answers[j]}\n"
                            
                    return complete_question, self.question_survey_id
            elif self.question_survey_id == self.n_questions_surveys -1:
                self.do_survey = False
                self.survey_answered = True
            else:
                return None
        else:
            return None
    
    def response_survey(self, answer_id, survey_id):
        self.question_survey_id = survey_id
        question, answers = self.conversation_analysis.obtain_question(self.question_survey_id)
        answer = answers[answer_id]
        # Save the response
        self.conversation_analysis.respond_question( number_question = self.question_survey_id, answer=real_answer)
        # Actualize the number of questions answered
        self.question_survey_id += 1
               
    def load_surveys(self,):
        user_path = self.user_manager.get_user_folder(self.user_manager.username, self.user_manager.password)
        # Survey path
        survey_path = os.path.join(user_path, 'surveys', 'surveys.csv')
        # Load the surveys
        self.n_questions_surveys = self.conversation_analysis.load_survey(path = survey_path)

    def save_surveys_responses(self,):
        self.conversation_analysis.save_survey_responses() 
        
    def load_description_database(self,):
        description = "Petroleum Geo-Services (PGS) is a leading provider of seismic and reservoir services for the oil and gas industry. The company offers a wide range of geophysical services, including seismic data acquisition, processing, and interpretation. PGS operates a fleet of seismic vessels and has a global presence, with offices and operations in various countries around the world. The company's services are used by oil and gas companies to identify and evaluate potential hydrocarbon reserves, optimize field development plans, and monitor reservoir performance. PGS is committed to delivering high-quality data and insights to its clients, helping them make informed decisions and maximize the value of their assets. "
        self.conversation_analysis.load_description_search_database(description = description)
        # Load prompt for the search database
        self.conversation_analysis.load_prompt_search_database(prompt = self.prompt_manager.get_prompt('search_in_database'))
    
    def response_with_audio(self, input_user, audio_flag=False, audio_response_flag = False):
        # Check if the input is a text or a audio
        if audio_flag:
            # If the input is a audio, we process the audio
            input_user = self.audio_transcriber.transcribe(input_user)
        # Obtain the response of Eve
        eve_message = self.response(input_user)
        
        if audio_response_flag:
            # If the response is a audio, we process the audio
            audio_path = os.path.join(self.temp_path, 'audio_responses')
            audio_path_file = self.audio_generator.create(eve_message, path_to_save = audio_path)
            return eve_message, audio_path_file
        else:
            return eve_message
        
          
    def response(self, message):
        ## Aca hay que chequear si se entra o no en la busqueda de la base de datos. 
        search_in_database = self.conversation_analysis.search_database(message)
        
        if search_in_database:
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)
            
            # Search in database
            eve_message = self.vectordabase_serch.predict(message)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)
        else:
            # We obtain the last messages
            last_messages = self.conversation_manager.get_n_last_messages()
            
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)

            # Obtain the complete current conversation
            current_conversation = self.conversation_manager.get_conversation()
            
            # Init the threads for the personal information and the summary conversation
            personal_info_thread = threading.Thread(target=self._process_personal_information, args=(current_conversation,))
            summary_thread = threading.Thread(target=self._process_summary_conversation, args=(last_messages , current_conversation))
            personal_info_thread.start()
            summary_thread.start()
            personal_info_thread.join()
            summary_thread.join()
            
            # Obtain the new personal information
            personal_info = self.conversation_analysis.get_personal_information()
            # Obtain the new summary conversation
            summary_conversation = self.conversation_analysis.get_summary_conversation()
            
            # Obtain the response of the chatbot
            eve_message = self.chatbot.response(message = message,
                                                personal_information = personal_info,
                                                previous_conversation_summary = summary_conversation,
                                                last_messages = last_messages)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)

        return eve_message
    
    def finalize(self,):
        # Update the personal information
        self.user_manager.update_user_information(key='personal_information', value=self.conversation_analysis.get_personal_information())
        
        # Save the conversation
        self.save_conversation()
        
        # Save the personal information 
        self.save_personal_information()
        
        # Save the surveys responses
        if self.survey_answered :
            self.save_surveys_responses()
            
            
class Eve_v6(Eve_v5):
    def __init__(self,):
        super().__init__()
        """Aca se cambio la forma en la cual se inicializa Eve"""
        # Initialize the conversation analysis
        self.conversation_analysis = ConversationAnalysis_v3()
        
        # Atributes for surveys
        self.n_questions_surveys = 0
        self.n_questions_surveys_answered = 0
        self.do_survey = False
        self.survey_answered = False
        
        self.temp_path = None
        self.username = None
        
    def initialize(self,username = None, password = None):
        # Login process
        self.user_manager.user_login(username = username, password = password)
        self.username = self.user_manager.get_username()
        
        # Load temp path
        self.load_tmp_file()
        
        # Load prompts
        self.prompt_manager.load_prompts(paths = [SUMMARY_CONVERSATION_v1['path'], PERSONAL_INFORMATION_v1['path'], EVE_PROMPT_v1['path'], SEARCH_IN_VECTORSTORE_DATABASE_v1['path']],
                                         keys = ['summary_conversation', 'personal_information', 'eve_message','search_in_database'],
                                         input_variables = [SUMMARY_CONVERSATION_v1['input_variables'], PERSONAL_INFORMATION_v1['input_variables'], EVE_PROMPT_v1['input_variables'], SEARCH_IN_VECTORSTORE_DATABASE_v1['input_variables']])
        
        # Load the information requiered for the conversation
        self.load_personal_information()
        
        # Load prompt for the summary conversation
        self.load_summary_conversation()
        
        # Load the description of the database
        self.load_description_database()
        
        # Load prompt for the chatbot
        self.chatbot.load_prompt(prompt = self.prompt_manager.get_prompt('eve_message'))
        

    def load_tmp_file(self,):
        self.temp_path = os.path.join('app','TEMP',self.username)
        # Check if path exists and if not create it
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)    
        # Folders to create
        folders_create = ['audio_responses']
        
        for folder in folders_create:
            # Check if path exists and if not create the folder
            if not os.path.exists(os.path.join(self.temp_path, folder)):
                os.mkdir(os.path.join(self.temp_path, folder))
        
    def ask_to_do_survey(self,):
        # Check if there is a survey to answer
        if self.n_questions_surveys > 0:
            # Generate the message of the survey
            message = "I have a survey for you. Do you want to answer it?(Y/N): "
            return  message
        
    def obtain_response_to_do_survey(self,response):
        response = response.upper()
        if response == "Y":
            self.do_survey = True
            self.question_survey_id = 0
        else:
            self.do_survey = False
            
    def get_survey(self, return_dict = False):
        # Check if there is a survey to answer
        if self.do_survey:
            # Check if question_survey_id < n_questions_surveys
            if self.question_survey_id < self.n_questions_surveys-1:
                if return_dict:
                    # Obtain the question
                    return self.conversation_analysis.obtain_question(self.question_survey_id, return_dict=return_dict)
                else:
                    question, answers = self.conversation_analysis.obtain_question(self.question_survey_id)
                    self.answers = answers
                    complete_question = question + " (choice: a,b,c,d,...)\n"
                    abcd = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',]
                    for j in range(len(answers)):
                        complete_question += f"{abcd[j]}. {answers[j]}\n"
                            
                    return complete_question, self.question_survey_id
                
            elif self.question_survey_id == self.n_questions_surveys :
                self.do_survey = False
                self.survey_answered = True
            else:
                return "None", "None"
        else:
            return "None", "None"
    
    def response_survey(self, answer_id, survey_id):
        self.question_survey_id = survey_id
        question, answers = self.conversation_analysis.obtain_question(self.question_survey_id)
        answer = answers[answer_id]
        # Save the response
        self.conversation_analysis.respond_question( number_question = self.question_survey_id, answer=answer)
        # Actualize the number of questions answered
        self.question_survey_id += 1
               
    def load_surveys(self, survey_id = None):
        user_path = self.user_manager.get_user_folder(self.user_manager.username, self.user_manager.password)
        # Survey path
        survey_path = os.path.join(user_path, 'surveys', 'surveys.csv')
        # Load the surveys
        self.n_questions_surveys = self.conversation_analysis.load_survey(path = survey_path, survey_id = survey_id)
        self.do_survey = True
        self.question_survey_id = 0
        
    def save_surveys_responses(self,):
        self.conversation_analysis.save_survey_responses() 
        
    def load_description_database(self,):
        description = "Petroleum Geo-Services (PGS) is a leading provider of seismic and reservoir services for the oil and gas industry. The company offers a wide range of geophysical services, including seismic data acquisition, processing, and interpretation. PGS operates a fleet of seismic vessels and has a global presence, with offices and operations in various countries around the world. The company's services are used by oil and gas companies to identify and evaluate potential hydrocarbon reserves, optimize field development plans, and monitor reservoir performance. PGS is committed to delivering high-quality data and insights to its clients, helping them make informed decisions and maximize the value of their assets. "
        self.conversation_analysis.load_description_search_database(description = description)
        # Load prompt for the search database
        self.conversation_analysis.load_prompt_search_database(prompt = self.prompt_manager.get_prompt('search_in_database'))
    
    def response_with_audio(self, input_user, audio_flag=False, audio_response_flag = False):
        # Check if the input is a text or a audio
        if audio_flag:
            # If the input is a audio, we process the audio
            input_user = self.audio_transcriber.transcribe(input_user)
        # Obtain the response of Eve
        eve_message = self.response(input_user)
        
        if audio_response_flag:
            # If the response is a audio, we process the audio
            audio_path = os.path.join(self.temp_path, 'audio_responses')
            audio_path_file = self.audio_generator.create(eve_message, path_to_save = audio_path)
            return eve_message, audio_path_file
        else:
            return eve_message
        
          
    def response(self, message):
        ## Aca hay que chequear si se entra o no en la busqueda de la base de datos. 
        search_in_database = self.conversation_analysis.search_database(message)
        
        if search_in_database:
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)
            
            # Search in database
            eve_message = self.vectordabase_serch.predict(message)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)
        else:
            # We obtain the last messages
            last_messages = self.conversation_manager.get_n_last_messages()
            
            # Add patient message to memory
            self.conversation_manager.add_message(message, is_ai_message = False)

            # Obtain the complete current conversation
            current_conversation = self.conversation_manager.get_conversation()
            
            # Init the threads for the personal information and the summary conversation
            personal_info_thread = threading.Thread(target=self._process_personal_information, args=(current_conversation,))
            summary_thread = threading.Thread(target=self._process_summary_conversation, args=(last_messages , current_conversation))
            personal_info_thread.start()
            summary_thread.start()
            personal_info_thread.join()
            summary_thread.join()
            
            # Obtain the new personal information
            personal_info = self.conversation_analysis.get_personal_information()
            # Obtain the new summary conversation
            summary_conversation = self.conversation_analysis.get_summary_conversation()
            
            # Obtain the response of the chatbot
            eve_message = self.chatbot.response(message = message,
                                                personal_information = personal_info,
                                                previous_conversation_summary = summary_conversation,
                                                last_messages = last_messages)
            
            # Add AI assistant message to memory
            self.conversation_manager.add_message(eve_message, is_ai_message = True)

        return eve_message
    
    def finalize(self,):
        # Update the personal information
        self.user_manager.update_user_information(key='personal_information', value=self.conversation_analysis.get_personal_information())
        
        # Save the conversation
        self.save_conversation()
        
        # Save the personal information 
        self.save_personal_information()
        
        # Save the surveys responses
        if self.survey_answered :
            self.save_surveys_responses()