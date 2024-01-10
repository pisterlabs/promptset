from openai import OpenAI


def gpt_pptmaker(topic, details, apikey):
    apikey = "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"
    client = OpenAI(
        api_key=apikey,
    )

    # prompt
    prompt = f"""
    make a powerpoint presentation about {topic} with the following details:
    {details}
    In the structure of:
    Slide 1:
    Title:
    Content:
    Slide 2:
    Title:
    Content:
    ...  
    Slide n:
    Title:
    Content:
    Only write the title of the slides and its content. 
    """

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are making ppt for the class, skilled in making index and its belongings.\
              showing only the title and content of the slides without other flowery words or summary or etc."},
            {"role": "user", "content": prompt}
        ],
    )
    # get format of messege contetn
    print(type(completion.choices[0].message.content))
    # get the response content only    
    return completion.choices[0].message.content



if __name__ == '__main__':
    topic = "What is the meaning of life?"
    details = "The meaning of life is to be happy and useful."
    apikey = "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"
    print(gpt_pptmaker(topic, details, apikey))
    
