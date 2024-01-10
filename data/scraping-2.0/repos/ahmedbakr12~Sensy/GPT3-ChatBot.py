
import openai as ai
import turtle
from transformers import pipeline, RobertaTokenizerFast, TFRobertaForSequenceClassification



def behavioral_analysis(question) :
    tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

    emotion_labels = emotion(question)
    emotion= emotion_labels[0]['label']
    return emotion


def chat(question,chat_log = None) -> str:
    if(chat_log == None):
        chat_log = start_chat_log
    prompt = f"{chat_log}Human: {question}\nAI:"
    response = completion.create(prompt = prompt, engine =  "davinci", temperature = 0.85,top_p=1, frequency_penalty=0, 
    presence_penalty=0.7, best_of=2,max_tokens=100,stop = "\nHuman: ")
    return response.choices[0].text

def modify_start_message(chat_log,question,answer) -> str:
    if chat_log == None:
        chat_log = start_chat_log
    chat_log += f"Human: {question}\nAI: {answer}\n"
    return chat_log

if __name__ == "__main__":


    ai.api_key = "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"

    completion = ai.Completion()

    start_chat_log = ""

    train = input("\nDo you want to train the openai chatbot (True/False): ")
    if(train == "True"):
        print("\n(To stop the training enter stop in the question)\n")
        while(True):
            question = input("Question: ")
            if question == "stop":
                break
            answer = input("Answer: ")
            start_chat_log = modify_start_message(start_chat_log,question,answer)
            print("\n")

    question = ""
    print("\nEnter the questions to openai (to quit type \"stop\")")
    while True:
        question = input("Question: ")
        print("\n emotion now: "+behavioral_analysis(question))



        if question == "stop":
            break
        ch=chat(question,start_chat_log)
        print("AI: "+ch)

        f= open("chatlist.txt","a")
        f.write("\n Human: "+question)
        f.write("\n emotion: "+behavioral_analysis(question))
        f.write("\n AI: "+ch)
        f.close()