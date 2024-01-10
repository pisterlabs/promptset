import openai

def answerQuestion(character, question):
    context = "Answer the question in {} style.\n\n".format(character)
    
    historyFile = open("history.txt", "a+")

    prompt = context + "Human: " + question + "\nAI: ".format(character)

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt = prompt,
        temperature=1,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        ).choices[0].text.replace("\n", "")

    historyFile.write(character + ": " + response + "\n\n")
    historyFile.close()

    return response