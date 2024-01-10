import re
import pandas as pd
import threading
import shelve
import json
import traceback
    
# ************************************************************
#       Class for OpenAI integration, contains different  
#       functionalities to submit requests and parse responses  
# ************************************************************ 

class AIQueries:
    def __init__(self, utils):
        self.db_lock = threading.Lock()
         
        # *************************************************************************** 
        #       Uncomment the code below in order to install the openai package
        # *************************************************************************** 
# Un comment OPENAI #        try:
# Un comment OPENAI #            import openai
# Un comment OPENAI #        except:
# Un comment OPENAI #            from pip._internal import main as pip
# Un comment OPENAI #            pip(['install', '--user', 'openai'])
# Un comment OPENAI #            import importlib
# Un comment OPENAI #            globals()['openai'] = importlib.import_module('openai')
# Un comment OPENAI #            import openai
# Un comment OPENAI #        self.oai = openai
        # *************************************************************************** 
        # for remote develop
        # Load your API key from an environment variable or secret management service
        self.oai.api_key = utils.get_configuration(key='AI-API-KEY.value')
        # for local develop
        # self.oai.api_key = 'AI-API-KEY'
        self.utils = utils

# *****************************************************************************************
#   parse GPT response     
# ******************************************************************************************  

        
    def clean_words_list(self, line:list):
        """remove (") from words in a list of words, keep (,) """
        def clean_word(word:str):
            found_special=False
            if word.strip()[-1]==',': 
                found_special=True 
                word = word[:-1]
            if word.strip()[-1]=='"':          
                word = word[:-1]
            if word.strip()[0]=='"':            
                word = word[1:]
            if found_special:
                word = word+','
            return word
        return [clean_word(word) for word in line]

    def clean_json_response(self, res_text:str):
        """Create DataFrame out of GPT's textual response (as JSON)
           The response might be in different (and sometimes not valid) Json formats 
           this function contains multiple tries to identify the proper Json format 
           and extract the values into a DataFrame
        """
        res_text = re.sub(r'\“',r'"',res_text)
        res_text = re.sub(r'\”',r'"',res_text)
        index_a = 0
        index_b = 0
        ### Trim trailing parenthesis
        try:
            index_a = res_text.index('{')
        except:
            True 
        try:
            index_b = res_text.index('[')
        except:
            True
        if index_a < index_b:
            res_text = res_text[index_a:]
        else:
            res_text = res_text[index_b:]
        ### Trim suffix parenthesis
        try:
            index_a = res_text.rindex('}')
        except:
            True 
        try:
            index_b = res_text.rindex(']')
        except:
            True
        if index_a > index_b:
            res_text = res_text[:index_a+1]
        else:
            res_text = res_text[:index_b+1]

        try :
            ### first try - load as valid json format to DataFrame
            df_result = pd.read_json(res_text)
        except:
            ### second try - load as list of values to DataFrame
            data = [res_text]
            df_result = pd.DataFrame(data=data, columns=['Could not parse result'])
            try:
                ### thired try - load as Json with index to DataFrame
                df_result = pd.read_json(res_text, orient='index')
                df_result.reset_index(inplace=True)
                df_result.columns  = ['key','value']
            except:
                try:
                    ### fourth try - apply some cleanup and adjustments to get a valid json format
                    res_text = re.sub(r'^([ \t]+)', '', res_text,flags=re.M)
                    clean_lines = ""
                    for i in res_text.splitlines():                                    
                        if i != "":
                            if i[0] not in '\"{}[]':
                                line = i.split(':')
                                if i[-1] == ',':
                                    i = '\"'+line[0]+'\"'+':'+'\"'+line[1][1:-1]+'\",'
                                else:
                                    i = '\"'+line[0]+'\"'+':'+'\"'+line[1][1:]+'\"'
                            else:
                                if ":" in i:
                                    line = i.split(':')                           
                                    words_list = self.clean_words_list(line[1:])
                                    line = [line[0]] + words_list
                                    if line[1] == " None":
                                        i = line[0] + ":" + "\"None\""
                                    if (not (len(line[1].strip()) == 1 and line[1].strip() in "\[\]{}")):
                                        if line[1][:2] != " \"" and line[1][0] != "\"":
                                            if i.strip()[-1] == ',':
                                                i = line[0]+':'+'\"'+line[1].strip()[:-1]+'\",'
                                            else:
                                                i = line[0]+':'+'\"'+line[1].strip()+'\"'

                            clean_lines += i
                    res_text = clean_lines
                    try:
                        res_json = json.loads(res_text)
                        if ("1" in res_json):
                            df_result = pd.read_json(res_text,orient="index")
                        else:
                            df_result = pd.read_json(res_text)
                    except:
                        df_result = pd.read_json(res_text)
                except Exception as e:
                    print(e)
                    print ("Could not parse results")
                    True
        return df_result


    def parse_respone(self,response:dict):
        """parse response from GPT API """

        res_text = response['choices'][0]['text']
        res_text = re.sub(r'^\d+:',r'',res_text.strip()).strip()
        if ": " in res_text:
            values = re.split(': ',res_text)
            if values[0] == 'Answer':
                return values[1]
            else:
                return values[0]
        else:
            return res_text
        
# *****************************************************************************************
#   generate query to GPT     
# ******************************************************************************************

    def ask_ai_with_cache(self, cache_filename:str, prompt:str):
        """create a request to GPT api, if the same question (prompt) already exist in cache,
           extract the response from cache 
        """
        index = 0
        with self.db_lock:
            db = shelve.open(cache_filename)
            if prompt in db:
                print ( 'Getting from cache')
                response = db[prompt]
                db.close()
                self.utils.add_to_query_log(prompt,re.sub(r'\n',r'<br>',response['choices'][0]['text']),'From cache')
                return response
            db.close()
            index = self.utils.get_query_log_size()
            self.utils.add_to_query_log(prompt,"Start Query",'No Cache')           
            self.utils.write_log_updates()
        max_token = 3900 - len(prompt)
        try:
            response = self.oai.Completion.create(
              model= "text-davinci-003",
              prompt= prompt,
              temperature=0.2,
              max_tokens=max_token,
              suffix=":",
              top_p=1.0,
              frequency_penalty=0.2,
              presence_penalty=0.0
            )
        except BaseException as error:
            self.utils.add_to_query_log(prompt,re.sub(r'\n',r'<br>',"Error: " + 'An exception occurred: {}'.format(error) + "\n" + traceback.format_exc()),'Fail', index)   
            response = {
                'choices': [
                    {
                        'text': "Error: " + str('An exception occurred: {}'.format(error))
                    }
                ]
            }
            return response
        with self.db_lock:
            db = shelve.open(cache_filename)
            db[prompt] = response
            db.close()
            self.utils.add_to_query_log(prompt,re.sub(r'\n',r'<br>',response['choices'][0]['text']),'No cache', index)  
        return response

    def get_item_from_list(self,item:str ,match_list:list):
        """use GPT's api to determine which of the elements in [match_list] is most similar to [item]"""             
        
        # if list contain only 1 value, skip the question
        if len(match_list)==1:
            return match_list[0]
        
        # if list contain the requested value (EQUAL)
        if item in match_list:
            return item
        
        # if list contain the requested value (EQUAL lower/upper)
        for list_item in match_list:
            if list_item.lower()==item.lower():
                return list_item
            
        # ask GPT to select the most similar value        
        question = "Please select from the following list the item which is equal (or the most similar) to \""+str(item)+"\"  :\n"
        for i in range(len(match_list)):
            question += str(i+1) + ": " + match_list[i] + "\n"
        question += "\nReplay only with the best value.\n"
        print (f"question used for matching elements: \n{question}")
        
        self.utils.add_time('Before ai get real '+str(item)+' match')
        response = self.ask_ai_with_cache('model_queries',question)
        self.utils.add_time('After ai get real '+str(item)+' match')

        res_text = self.parse_respone(response)
        print(res_text)
        try:
            i = int(res_text) -1
            return match_list[i]
        except:
            for i in range(len(match_list)):
                if res_text.strip().lower() == match_list[i].lower():
                    return match_list[i]
        return "Unkonwn"
