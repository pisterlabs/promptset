from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from .translate import Chatbot_translator_EN_to_PL, Chatbot_translator_PL_to_EN
from os.path import dirname, join 
from dotenv import load_dotenv
from .vecdb import cossimhist, retreive_hist 
# from .translate_deepl import translate_to_pl, translate_to_en
from .fake_agents import *
# from .survey_analysis import NotesAnalyst

#.env adjustments
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_TOKEN = os.environ.get("APIKEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
SYSMSG_PARENTS = os.environ.get("SYSMSG_PARENTS")
SYSMSG_HAPPY = os.environ.get("SYSMSG_HAPPY")
SYSMSG_SAD = os.environ.get("SYSMSG_SAD")  
SYSMSG_ANGRY = os.environ.get("SYSMSG_ANGRY")  
SYSMSG_FRIENDLY = os.environ.get("SYSMSG_FRIENDLY")  
SYSMSG_SUICIDE = os.environ.get("SYSMSG_SUICIDE")  


#chatbot class
class ChatbotWithHistory:
    def __init__(self, is_for_kids: bool, emotion:str):
        if is_for_kids:
            if emotion == 'HAPPY':
                self.template = SYSMSG_HAPPY
            elif emotion == 'SAD':
                self.template = SYSMSG_SAD
            elif emotion == 'ANGRY':
                self.template = SYSMSG_ANGRY
            elif emotion == 'FRIENDLY':
                self.template = SYSMSG_FRIENDLY
        else:
            self.template = SYSMSG_PARENTS
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=self.template
        )
        GenParams().get_example_values()
        #model hyperparameters 
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 10,
            GenParams.MAX_NEW_TOKENS: 150,
            GenParams.TEMPERATURE: 0.0,
            GenParams.REPETITION_PENALTY: 1,
            GenParams.LENGTH_PENALTY: {'decay_factor': 2, 'start_index': 90}
        }
        #initializing the model 
        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": API_TOKEN,  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=PROJECT_ID
        )
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            input_key='human_input'
        )
        self.chain = LLMChain(llm=self.model.to_langchain(), prompt=self.prompt, verbose=False, memory=self.memory)
        self.ptoe = Chatbot_translator_PL_to_EN()
        self.etop = Chatbot_translator_EN_to_PL()
    #a method to get the model's response to some prompt + history 
    def get_response(self, inp: dict):
        alert_state = inp['alert_state']
        if alert_state:
            self.template = SYSMSG_SUICIDE
        last_prompt_str_pl = inp['new_prompt']['prompt'] #str of the last prompt
        lps_en = self.ptoe.get_translation_ptoe({'human_input':last_prompt_str_pl})
        print(lps_en)
        # lps_en = translate_to_en(last_prompt_str_pl)
        last_prompt_emb = inp['new_prompt']['vectorized_prompt'] #embedding of the last prompt
        prompt_formatted_str = self.template.format(person_summary=None, chat_history=None, human_input=lps_en)
        #handling an empty database 
        if len(inp['history']) > 0:
            prev_prompts = retreive_hist(inp)
            #running cosine similarity on the entire chat history to retreive the most relevant messages
            n_prompts_answers = cossimhist(last_prompt_emb, vec_dict=prev_prompts)
            prompt_formatted_str = self.prompt.format(chat_history=n_prompts_answers, human_input=lps_en)
            
            response_en = self.chain({'person_summary':prompt_formatted_str, 'human_input':lps_en})
            print(response_en)
            response_pl = self.etop.get_translation_etop({'human_input':response_en})
            print(response_pl)
            # response_pl = translate_to_pl(response_en)
        else:
            prompt_formatted_str = self.template.format(person_summary=None, chat_history=lps_en, human_input=lps_en)
            response_en = self.chain({'person_summary':None, 'human_input':lps_en})
            print(response_en)
            response_pl = self.etop.get_translation_etop({'human_input':response_en})
            print(response_pl)
            # response_pl = translate_to_pl(response_en)
        return response_pl