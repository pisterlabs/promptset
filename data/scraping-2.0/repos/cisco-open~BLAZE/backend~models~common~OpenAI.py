import openai
from flask import current_app

def get_openAI_info():
    """ 
    Function to return a dictionnary containing the name, class name, 
    description, paper link and GitHub repo link of the BART model. It is used 
    throughout the code to get various information about the model.

    Returns
    -------
    model_info : a dictionnary
        A dictionnary containing the name, class name, 
        description, paper link and GitHub repo link of the T5 model
    """
    model_info = {
        'name': "OpenAI",
        'class_name': 'OpenAI',
        'desc': "OpenAI",
    }
    return model_info

class OpenAI():
    tasks_supported = ["actionables","summarization"]

    def __init__(self):
        self._info = get_openAI_info()
    
    def _get_model_info(self):
        pass

    def _get_name(self):
        return self._info['name']

    def _get_class_name(self):
        return self._info['class_name']
    
    def gpt_analysis(self, category, processed_text, prompt=None): 
        
        print("Reached GPT analysis")
        #return {'choices' : [{'text': "DUMMY RESPONSE"}]}
        if prompt is not None:
            message = f"{prompt}\n{processed_text}"
        else:
            if category == "summary": 
                print("Coming to summarize")
                prompt = "Analyze the following meeting transcript and generate a summary."
                message = f"{prompt}\n{processed_text}"
            elif category == "actionables": 
                prompt = "Analyze the following meeting transcript and identify actionable items (such as todo's) and return them in a list, separated by the pipeline '|' character" 
                message = f"{prompt}\n{processed_text}"
                print(message)
            elif category == "agenda": 
                prompt = "Analyze the following meeting transcript and idetnify discussed topics as well as the duration they were discussed and return them in a list, separated by the '-' between time and label, and separated by the pipeline '|' character between each item. For example, 'XX:XX - Introductions' may be a valid entry in the returned list, if the meeting contained an introduction." 
                message = f"{prompt}\n{processed_text}"
            else: 
                return None  

    
        openai.api_key = current_app.config.get('OPENAPI_KEY')
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=message,
            temperature=0.7,
            max_tokens=892,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response 

    
    def _summarize_text(self, text_to_summarize):
        response = self.gpt_analysis("summary",text_to_summarize)
        return response['choices'][0]['text']
    
    def get_actionables(self,text):
        response = self.gpt_analysis("actionables",text)
        return response['choices'][0]['text']