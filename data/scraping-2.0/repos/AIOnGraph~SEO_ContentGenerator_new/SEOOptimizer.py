from openai import OpenAI


def initilize_openai(OPEN_AI_API):
    open_ai = OpenAI(api_key=OPEN_AI_API)
    return open_ai

def convert_temperature_from_string_to_int(temperature_value):
    print(temperature_value)
    temperature_for_openai=0.0
    if temperature_value =="Different Ideas":
        temperature_for_openai=0.3
    elif temperature_value=="Out Of Ordinary Ideas":
        temperature_for_openai=0.6
    elif temperature_value=="Wild Idea":
        temperature_for_openai=0.9
    return  temperature_for_openai
        
        
   
def topic_generator(input_text,temperature_value,topic_language,topic_type,OPEN_AI_API):
    open_ai = initilize_openai(OPEN_AI_API)
    temperature_value_in_number=convert_temperature_from_string_to_int(temperature_value)
    prompt = f"""topic text is {input_text},topic output language is {topic_language} and this topic will be used for {topic_type}
""" 
    messages=[
    {"role": "system", "content": "You are a helpful assistant who will give 5 SEO optimized Topic about given topic text by using given information."},
    {"role": "user", "content": prompt}]
    response=open_ai.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temperature_value_in_number,messages=messages,)
    
    res=response.choices[0].message.content.split("\n")
    while("" in res):
        res.remove("")
    for i in range(len(res)):
        res[i] = res[i].replace('"', '')
    print(res)
    return res
    

def content_generator(content_topic,content_type,content_length,focus_market,content_language,audience_type,OPEN_AI_API):
    open_ai = initilize_openai(OPEN_AI_API)
    prompt = f"""content topic is {content_topic},Content type is {content_type} and content length is {content_length},main focused market is of {focus_market} country,type of audience {audience_type},output response will be in {content_language}language.
"""  
    messages=[
    {"role": "system", "content": "You are a helpful assistant who will create  a content for a given type by following given instructions."},
    {"role": "user", "content":prompt }]
    
    response=open_ai.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,messages=messages,)
    return response.choices[0].message.content
    
 