# Mock Functions - for testing driver code
import random
import openai
import json
from emote_behavior import MISTY_ARM_ACTIONS,MISTY_HEAD_ACTIONS,MISTY_FACE_ACTIONS

# Set up: OpenAI API key + Misty's Actions
api_key = "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"  # Replace 'YOUR_API_KEY' with your actual API key
openai.api_key = api_key
actions = list(MISTY_ARM_ACTIONS.keys()) + list(MISTY_HEAD_ACTIONS.keys()) + list(MISTY_FACE_ACTIONS.keys())

# Prompting Constants 
follow_up_prompt = "(1) provide response to user and ask follow up question. (2) choose the appropriate movements for the robot to take from the following set of actions: {}. Return response in JSON format with 'response' and 'movements' as keys, without any deviation.".format(actions)
end_convo_prompt = "(1) provide response to user and end the conversation. (2) choose the appropriate movements for the robot to take from the following set of actions: {}. Return response in JSON format with 'response' and 'movements' as keys, without any deviation.".format(actions)


###### FUNCTION DEFINITIONS ####################################################
def _chatgpt_interface(chat_history):
    response = openai.chat.completions.create( 
                        model="gpt-4", # previously used "gpt-3.5-turbo"
                        messages = chat_history,
                        temperature = 0.7, 
                        max_tokens = 150
                    )
        
    # Return AI's response - JSON {"response": _ , "movements": _}
    return response.choices[0].message.content

def prompt_openAI(user_input, response_type="follow_up", sentiment="N/A", chat_history=[]): 
    # Engineer the Prompt
    message = ""
    if response_type == "follow_up":
        message = {"role": "user", "content": f"[sentiment: {sentiment}, user input: {user_input}]: {follow_up_prompt}"}
    else:
        message = {"role": "user", "content": f"[sentiment: {sentiment}, user input: {user_input}]: {end_convo_prompt}"}
    chat_history.append(message)
    
    # Get Chat GPT Response + Update Chat History 
    init_response = json.loads(_chatgpt_interface(chat_history))
    response, movements = [init_response["response"], init_response["movements"]]
    chat_history.append({"role": "system", f"content": response})

    return response, movements, chat_history

if __name__ == "__main__":
    print(follow_up_prompt)