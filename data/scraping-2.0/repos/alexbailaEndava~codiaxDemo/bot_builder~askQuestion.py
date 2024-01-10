import openai

def askQuestion():
    question = input()
    historyFile = open("history.txt", "a+")
    historyFile.write("Human: " + str(question) + "\n")
    return question