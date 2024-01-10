import openai
import json
import mailsend
import smsSend

def recommend_chat(result_text,final_mail_id,ph_no):
    openai.api_key = "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"  
    # print(result_text)
# Here we are declaring a variable "chat" what holds the voice commands generate text
    chat = result_text

    
    respond = openai.Completion.create(model="text-davinci-003",
 
    
                                    
    prompt = """Take the chat and Based on the provided values, give detailed personalized skin care suggestions: 
    - suggest recommendation based on if wrinkles are found or not
    - suggest recommendation based on temperature and humidity of place user lives in
    - suggest recommendation based on contrast of skin whether low, medium, high
    - suggest recommendation based on pigmentation of skin found low, medium, high
    - suggestions should be purely based on temperature and humidity taking into consideration



    Text:  """+chat+"""

    
    """ ,
                               

                                    
    temperature=0, max_tokens=1000,

    top_p=1,

    frequency_penalty=0.2,

    presence_penalty=0 )


# Extract the generated text from the response
    generated_text = respond.choices[0].text.strip()

# Display the generated text as output
    print(generated_text)
    # print("recommend: ",final_mail_id,result_text,generated_text,ph_no)
    mailsend.send_mailToCustomer(final_mail_id,result_text,generated_text)
    smsSend.send_sms(ph_no,result_text,generated_text)

