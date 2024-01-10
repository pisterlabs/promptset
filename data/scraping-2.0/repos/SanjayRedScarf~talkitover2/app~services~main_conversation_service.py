from flask import session
from services import string_cleansing_service, trigger_check_service, trigger_response_service, gpt3_service
from repositories import triggers_repository
from threading import Thread
from models import output_data
import ast
import openai

triggers_that_can_be_repeated = ["msgIsQuestion"]
gpt3 = gpt3_service.gpt()

class MainConversationService:

    def __init__(self, next_user_input_option_types):
        self.next_user_input_option_types = next_user_input_option_types

    def get_main_conversation_output_data(self, conversation_input_data, user_character_count,sentence_encoder):
        
        _string_cleansing_service = string_cleansing_service.StringCleansingService()
        _trigger_check_service = trigger_check_service.TriggerCheckService()
        _trigger_response_service = trigger_response_service.TriggerResponseService()
        _trigger_repository = triggers_repository.TriggersRepository()

        cleaned_message = _string_cleansing_service.clean_string(conversation_input_data.message)

        trigger = _trigger_check_service.get_trigger(cleaned_message, conversation_input_data.message)
        response = _trigger_response_service.get_response_for_trigger(cleaned_message, trigger, user_character_count)            
        #session['response_type'] = trigger
        ai_data ={} # look into getting rid of this if possible, maybe in ouptup_data.py
        if response in session['TRIGGERS_DICT']["encouragingNoises"]['triggers']:
            try:
                #ai_data = sentence_encoder.get_cat(cleaned_message) # should this be cleaned?
                ai_data = sentence_encoder.get_cat_no_cut(conversation_input_data.message)
                in_cats = ai_data['max_over_thresh']
                trigger, response = sentence_encoder.guse_response(in_cats,session['ai_repeat'])
                if trigger == None:
                    trigger = 'encouragingNoises'
                    #session['response_type'] = trigger
                    response = _trigger_response_service.get_response_for_trigger(cleaned_message, trigger, user_character_count)
                if trigger != 'encouragingNoises':
                    trigger += '_ai'
            except Exception as e:
                print(e)
                trigger = 'encouragingNoises'
                #session['response_type'] = trigger
                response = _trigger_response_service.get_response_for_trigger(cleaned_message, trigger, user_character_count)

        # Remove from dictionary as response has now been used and we do not want it to be repeated.
        if (type(trigger) != dict) & (trigger not in triggers_that_can_be_repeated)&(trigger[-3:] != "_ai"):
            Thread(target=_trigger_repository.remove_used_trigger(trigger)).start()

        next_user_input = self.next_user_input_option_types.next_user_input_free_text
        
        # this is here so that information is written accuratly in the stored data, has no effect on the response outputted 
        if trigger == 'encouragingNoises' and response not in session['TRIGGERS_DICT']["encouragingNoises"]['triggers']:
            trigger = 'specialCase'
        
        
        if session['gpt3']:
            try:
                response = gpt3.get_response(cleaned_message)['choices'][0]["text"].lstrip().split('\nHuman:')[0].lstrip("\"\'")
                trigger = 'gpt3'
            except:
                session['gpt3'] = False
        
        session['last_trigger'] = trigger

        return output_data.OutputData(response, conversation_input_data.section, [""], next_user_input, "freeText",ai_data,trigger)

        