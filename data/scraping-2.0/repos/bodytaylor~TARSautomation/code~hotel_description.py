import openai
openai.api_key = "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"

def comp(PROMPT, MaxToken=50, outputs=3): 
    # using OpenAI's Completion module that helps execute  
    # any tasks involving text  
    response = openai.Completion.create( 
        # model name used here is text-davinci-003 
        # there are many other models available under the  
        # umbrella of GPT-3 
        model="text-davinci-003", 
        # passing the user input  
        prompt=PROMPT, 
        # generated output can have "max_tokens" number of tokens  
        max_tokens=MaxToken, 
        # number of outputs generated in one call 
        n=outputs 
    ) 
    # creating a list to store all the outputs 
    output = list() 
    for k in response['choices']: 
        output.append(k['text'].strip()) 
    return output


PROMPT = """ 

Shorten this text to fit 250 characters,  do not change the words or change sentence structure, do not include special characters like " / & ; @ % ' 
Text:### 
text: Experience urban tranquility at Novotel Singapore On Kitchener, your haven in the heritage hub of Little India. Wander Serangoon Road, where vibrant hues and alluring aromas enchant. Dive into the 24-hour shopping at Mustafa Centre. After indulging in Singapore's sensations, relax in lavish modernity within your contemporary furnished room. Our devoted team is poised to lead you in embracing the local essence, ensuring an unforgettable stay.
###
"""
print(comp(PROMPT, MaxToken=3000, outputs=1))