import openai 
openai.api_key = "your-api-key" 
def translate_text(text, target_language): 
    response = openai.Completion.create( 
    
    engine="text-davinci-002", 
    prompt=f"Translate the following text into {target_language}: {text}\n", 
    max_tokens=60, 
    n=1, 
    stop=None, 
    temperature=0.7, ) 
    return response.choices[0].text.strip()
    


text = "Zdravo svete" 
target_language = "englsih"  
translation = translate_text(text, target_language) 
print(translation) 

