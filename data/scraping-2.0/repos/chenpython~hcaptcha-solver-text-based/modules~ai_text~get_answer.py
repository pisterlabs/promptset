import os
import openai
import time

f = open('./data/api-secret.txt', 'r')
openai.api_key = f.readline()

def store_answers(text):
    with open("./data/answers.txt", "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0 :
            file_object.write("\n")

        file_object.write(text)


def openai_create(prompt):
    while True:
        try:
            response = openai.Completion.create(
            ## Works the similar to 003, costs less
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            temperature=0.5,

            ## If you felt like ai was giving alot of wrong answers just uncomment this part.
            ## Remmber using this model will cost you more. -- Check free tier openai for prices            
            # model="text-davinci-003",
            # prompt=prompt,
            # temperature=0.9,
            # max_tokens=150,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0.6,
            # stop=[" Human:", " AI:"]
            )
            temp_ans = response.choices[0].text
            if  temp_ans != '':
                break            
        except:
            print("Error at Request.\nGoing to sleep: 30 Seconds!")
            time.sleep(30)
    return response.choices[0].text



def chatgpt_clone(input,history):

    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = openai_create(inp)
    history.append((input, output))
    return history, history


history = []
def find_answer(question):
    global history

    #if you have OpenAI API key as a string, enable the below
    
    
    input_text = question

    history, output = chatgpt_clone(input_text,history)
    answer = output[-1][1]
    
    final_final_answer = ""
    if 'no' in answer.lower():
        question = question.replace(' Give answer in format as "Yes" or "No"','')
        final_final_answer = 'no'
        text = question+'|'+'no'
        store_answers(text)

    elif 'yes' in answer.lower():
        question = question.replace(' Give answer in format as "Yes" or "No"','')
        final_final_answer = 'yes'
        text = question+'|'+'yes'
        store_answers(text)

    return final_final_answer








