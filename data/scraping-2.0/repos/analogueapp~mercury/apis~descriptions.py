import openai
import os

openai.api_key = os.getenv("OPENAI_KEY")

def get_description(work, creatorList):  
    prompt_text = (f"Generate a description for {work}, by {', '.join(creatorList)}. Simply print 'Unknown' if unfamiliar with the work")
    completion = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt_text, max_tokens=3000, temperature=0.6, n=1)       
            
    description = completion.choices[0].text.strip().strip('"')    
     
    if description == "Unknown":
        return None
    else:
        return description