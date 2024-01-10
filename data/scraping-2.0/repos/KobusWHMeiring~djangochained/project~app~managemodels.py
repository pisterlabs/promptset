from app import openai, hfcontroller


def sendModel(user_message, system_message, transcript, model):
    
    if model == "gpt3.5": 
        response_data = openai.chat3turbo(user_message, system_message, transcript)
    elif model == "gpt3":
        response_data = openai.chat3turbo(user_message, system_message, transcript)
    elif model == "google_flan":
        response_data = hfcontroller.flant5(user_message)
    else:
        print("failed")
        

    return response_data        