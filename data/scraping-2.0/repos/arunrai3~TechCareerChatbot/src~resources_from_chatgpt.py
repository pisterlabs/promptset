import openai

#--------------------------
#resources_from_chatgpt
#--------------------------

def resourcesAtTheEnd(career):    
    """
    
    This function presents a question as well as multiple choice options to select from that help user select a career in tech.

    :param career (str): The career ChatGPT is going to find additional resources for.

    :return: The resources returned from ChatGPT
    :rtype: str   
    
    """
    
    openai.api_key = ""
    
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are helping with a chatbot, do not add any extra text except for bullet points."},
        {"role": "user", "content": r"Give me a list of bullet points of tip and resources for the following career: " + career}
      ]
    )
    resources = completion.choices[0].message
    return resources
