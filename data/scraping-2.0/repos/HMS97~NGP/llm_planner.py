from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
import re
import numpy as np


def llm_choose_database(question):


    template = """"Analyze the given statement or word: {question} and determine which of the following databases is most suitable for answering the question:
    if the question is like someone says someting, choose speech_data.\
    if the question is like someone does something, or take some actions, choose vision_data.\
    if the question is like other situation invole both vision and speech, choose summary_data.\
    'vision_data': Contains text data obtained from image or video captions.
    'speech_data': Contains text data obtained through speech recognition.
    'summary_data': Contains a comprehensive summary of a video, including both vision and speech text data.
    You are required to return one dataset's name which is the most suitable database without explanation. Choose only one from vision_data, speech_data, or summary_data.
    """
    PROMPT = PromptTemplate(template=template, input_variables=["question"])
    chain = LLMChain(llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo'), prompt=PROMPT)
    result = chain.predict(question = question)

    return result


def llm_check_relevent_text(question, Given_text):

    template = """Please analyze the following text and identify the frame intervals that are directly related to the word {question}. \
    Return the frame numbers in JSON format as follows:  [(start, end), ...] .  \ 
    If none of the frame intervals are related to the word {question}, 
    return [ ] . Note: The return frame interval's text no explanations or assumptions should be made.
        
        
    {Given_text}
    """
    PROMPT = PromptTemplate(template=template, input_variables=["question", "Given_text"])
    chain = LLMChain(llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo'), prompt=PROMPT)
    result = chain.predict(question = question, Given_text = Given_text)

    try:
        result = eval(result)
        if len(result) == 0:
            result = []
    except:
        print(result)
        print('something wrong, return []')
        result = []
    return result


def llm_doule_check_relevent_text(question, Given_text):

    template = """Please analyze the following text and identify the scene  really happen with  {question}. \
    There may have some noise in the given text, I have add more close context to help you. \
    please analsis the vison and speech text to judge if the frame is really about to {question}. \
    No explanation! Output format [Yes] or [No]. if you are not sure, return [No]. \
   
    {Given_text}
    """
    PROMPT = PromptTemplate(template=template, input_variables=["question", "Given_text"])
    chain = LLMChain(llm = OpenAI(temperature=0.5, model_name='gpt-3.5-turbo'), prompt=PROMPT)
    result = chain.predict(question = question, Given_text = Given_text)
    # print(result,Given_text)
    if 'No' in result :
        return False
    else:
        return True
    

def llm_doule_add_relevent_text(question, Given_text):
    template = """Pretend to be GPT4. Please analyze the following text. \
    There may have some noise in the given text, I have add more close scenes context to help you. \
    please analsis the vison and speech text to judge if there are some frames or objects in frames can be treated scenes about {question} or could have indirect relations with {question}. \
    No explanation! Output format:[Yes] or [No]. [Yes] 
    if may have  a chance and  if you are not sure., return [Yes] 
    if you are confidence with there isn't, return [No] \

    {Given_text}
    """
    PROMPT = PromptTemplate(template=template, input_variables=["question", "Given_text"])
    chain = LLMChain(llm = OpenAI(temperature=0.9, model_name='gpt-3.5-turbo'), prompt=PROMPT)
    result = chain.predict(question = question, Given_text = Given_text)
    print(result)
    if 'No' in result :
        return False
    else:
        return True
    
    
    
def llm_similar_words(question, words_number= 5):

    # template = """ Return a list of words   of length {words_number} that share a semantic relationship with  {question}  in  word embedding vector space.\
    # only return a string list including  {question} at begain, no explanation. 
    # """         if question is a name, return other name with that person.\

    template = """ Given a {question}, return words or sentence are  similar to {question} in the word embedding vector space.\
        The output should  be a list, like [wrods_1, words_2] The list's length is {words_number}, with NO additional explanation.
    """
    
    PROMPT = PromptTemplate(template=template, input_variables=["question","words_number"])
    chain = LLMChain(llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo'), prompt=PROMPT)
    result = chain.predict(question = question, words_number = words_number)
    def remove_newline_number(string):
        return re.sub(r'\n\d+', '', string)


    try:
        # print(result.split(','))
        result = result.split('.')
        result.append(question)
        if question.isalpha() and not question.endswith("ing"):
            result.append(question+'ing')

        result = [remove_newline_number(i) for i in result]
        result = [i.strip() for i in result if len(i)>2]
    except:
        result = [question]
        print('no similar words, return question')
    result = np.unique(result)
    return result