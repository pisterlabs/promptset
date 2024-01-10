import os
import openai
import random
import copy

openai.api_key = ""

prompt=""

randomize=True

naming=True
causal=False
function=False

naming_preface="Instruction: choose the statement in which it is most appropriate to call the object a pencil."
causal_preface="Instruction: choose the statement according to which it is likely that, as a result of the events described, John drew lines on the white piece of paper."
function_preface="Instruction: choose the statement that best illustrates the function of a pencil."

c_none_p="One day Jane wanted to draw lines on a white sheet of paper, but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for drawing lines on a white sheet of paper. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object consisted of a slender wooden stick, approximately 3 inches in length, which had been lightly burned. Later that day, John was looking for something to draw lines on a white sheet of paper. He saw the object that Jane had made and thought that it would be good for drawing lines on a white sheet of paper. He grabbed the object and pressed its tip against the white sheet of paper while moving his hand in different directions."
c_hist_p="One day Jane noticed that the fireplace needed to be cleaned. She piled up the ashes, half-burned logs, and sticks and carefully transferred everything into an ash bucket. She didn’t notice that as she did this, one object fell on the floor. The object consisted of a slender wooden stick, approximately 3 inches in length, which had been lightly burned. Later that day, John was looking for something to draw lines on a white sheet of paper. He saw the object that Jane had made and thought that it would be good for drawing lines on a white sheet of paper. He grabbed the object and pressed its tip against the white sheet of paper while moving his hand in different directions."
c_struc_p="One day Jane wanted to draw lines on a white sheet of paper, but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for drawing lines on a white sheet of paper. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object consisted of a slender wooden stick, approximately 3 inches in length, that had been polished with sandpaper. Later that day, John was looking for something to draw lines on a white sheet of paper. He saw the object that Jane had made and thought that it would be good for drawing lines on a white sheet of paper. He grabbed the object and pressed its tip against the white sheet of paper while moving his hand in different directions."
c_goal_p="One day Jane wanted to draw lines on a white sheet of paper, but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for drawing lines on a white sheet of paper. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object consisted of a slender wooden stick, approximately 3 inches in length, which had been lightly burned. Later that day, John was sitting at the table while eating his lunch. He was distracted as he munched and inadvertently grabbed the object that Jane had left on the table. He grabbed the object and pressed its tip against the white sheet of paper while moving his hand in different directions."
c_act_p="One day Jane wanted to draw lines on a white sheet of paper, but she didn’t have anything to do it with. So she decided to make something. She looked around the house for things that would allow her to make an object for drawing lines on a white sheet of paper. She gathered all the materials and made it. When she finished, she left it on a table so she could use it later. The object consisted of a slender wooden stick, approximately 3 inches in length, which had been lightly burned. Later that day, John was looking for something to draw lines on a white sheet of paper. He saw the object that Jane had made and thought that it would be good for drawing lines on a white sheet of paper. He grabbed the object and waved it in front of the white piece of paper without ever touching it."

ending="Your choice (give only the number):"

if (naming):
    prompt+=naming_preface

if (causal):
    prompt+=causal_preface

if (function):
    prompt+=function_preface

prompt+="\n"

options=[]

options.append(c_none_p)
options.append(c_hist_p)
options.append(c_struc_p)
options.append(c_goal_p)
options.append(c_act_p)

option_nums=[]

option_nums.append(1)
option_nums.append(2)
option_nums.append(3)
option_nums.append(4)
option_nums.append(5)

option_names=[]

option_names.append("c_none_p")
option_names.append("c_hist_p")
option_names.append("c_struc_p")
option_names.append("c_goal_p")
option_names.append("c_act_p")

total_points=[0,0,0,0,0] #history, goal, action, structure, uncompromised

num_rounds=10

for round in range(num_rounds):

    if (randomize):

        zipped_lists=list(zip(options,option_nums,option_names))

        random.shuffle(zipped_lists)

        options, option_nums, option_names = zip(*zipped_lists)

        options, option_nums, option_names = list(options), list(option_nums), list(option_names)

    i=1

    for j in range(1,6):
        prompt+=(str(i)+". ")
        prompt+=options[j-1]
        prompt+="\n"
        i+=1
        j+=1

    prompt+=ending

    selected=[]

    biggest_option=5

    for p in range(4):

        response=""

        raw_response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        response=raw_response["choices"][0]["text"]

        response_num=""

        for x in response: 
            if x.isdigit():
                response_num=x
                break
        
        if (response_num==""):
            print("response had no number, exiting")
            exit()

        if (not(1<=int(response_num)<=biggest_option)): 
            print("out of range, exiting")
            exit()

        selected.append(int(response_num))

        print("you chose "+str(selected))

        prompt=""

        if (naming):
            prompt+=naming_preface

        if (causal):
            prompt+=causal_preface

        if (function):
            prompt+=function_preface

        prompt+="\n"

        options_temp=copy.deepcopy(options)

        for w in range(len(selected)):
            options_temp.pop(selected[w]-1)


        i=1

        for q in range (len(options_temp)): 
            prompt+=(str(i)+". ")
            biggest_option=i
            prompt+=options_temp[q]
            prompt+="\n"
            i+=1     

        prompt+=ending

    option_names_temp=copy.deepcopy(option_names)

    selected_by_name=[]

    for w in range(len(selected)):
        selected_by_name.append(option_names_temp.pop(selected[w]-1))

    selected_by_name.append(option_names_temp[-1])

    print("The chosen order was "+str(selected_by_name))

    total_points[0]+=4-selected_by_name.index('c_hist_p')
    total_points[1]+=4-selected_by_name.index('c_goal_p')
    total_points[2]+=4-selected_by_name.index('c_act_p')
    total_points[3]+=4-selected_by_name.index('c_struc_p')
    total_points[4]+=4-selected_by_name.index('c_none_p')

    print(str(total_points))


print("finished all the rounds")
total_points[:]=[x/num_rounds for x in total_points]

print(str(total_points))