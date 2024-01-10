import os
import openai as ai

ai.api_key = "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"

gptResult = None

def generate_gpt3_response(print_output=False):
    user_text = input("What do you want to search for?")
    
    completions = ai.Completion.create(
        engine='text-davinci-003',  
        temperature=0.5,            
        prompt=user_text,           
        max_tokens=100,             
        n=1,                        
        stop=None,                  
    )

    if print_output:
        print(completions)
    
    return completions.choices[0].text
