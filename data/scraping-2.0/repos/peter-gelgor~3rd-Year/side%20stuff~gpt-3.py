import os
import openai

openai.api_key = "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"

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

def conversation(humanPrompts, aiPrompts):
    if (len(humanPrompts) == 0):
        humanPrompts.append(input("please put the first human question: "))
    if (len(aiPrompts) == 0):
        aiPrompts.append(input("please put the first ai response for training: "))
        if (len(humanPrompts) == 1):
            humanPrompts.append(input("please put the second human question: "))

    if (len(humanPrompts) == len(aiPrompts)):
        humanPrompts.append(input("please put the next human question: "))

    tinderPrompt = "The following is a conversation with a potential romantic interest. The AI in question is creative, clever, inquisitive, and very friendly. This AI will try to get the human to go on a date with them\n\n"
    for i in range(len(humanPrompts)):
        if (i == len(humanPrompts) - 1):
            tinderPrompt += "Human:" + humanPrompts[i] + "\n" + "AI:"
        else:
            tinderPrompt += "Human:" + humanPrompts[i] + "\n" + "AI:" + aiPrompts[i] + "\n"
        
    response = openai.Completion.create(
        engine="davinci",
        prompt=tinderPrompt,
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    aiResponse = response["choices"][0]["text"]
    print("AI: " + aiResponse)
    # print(len(aiPrompts))
    aiPrompts.append(aiResponse)
    # print(len(aiPrompts))
    conversation(humanPrompts, aiPrompts)


conversation(
    ["What do you think of my skateboard?", "Thank you!! Ahh no that sucks come borrow mine :)"], 
    
    ["Sick skateboard!! Mine got totalled by a car a few days ago :("])