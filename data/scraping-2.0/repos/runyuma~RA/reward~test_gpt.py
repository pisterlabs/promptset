import openai
key = "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"
openai.api_key = key
lang_goal = 'put the yellow block on a blue bowl'

obj_bp = [['green bowl', [63, 174],[20,20], 0.34195197],
                ['yellow bowl', [58, 30],[20,20], 0.32001296], 
                ['blue bowl', [175, 132],[20,20], 0.30598456], 
                ['yellow block', [79, 119],[5,5], 0.27325985], 
                ['green block', [129, 181],[5,5], 0.27240032], 
                ['blue block', [173, 47],[5,5], 0.26218402]]
obj_ap = [['green bowl', [63, 174],[20,20], 0.34195197],
                ['yellow bowl', [58, 30],[20,20], 0.32001296], #
                ['blue bowl', [175, 132],[20,20], 0.30598456], 
                ['yellow block', [79, 119],[5,5], 0.27325985], 
                ['green block', [129, 181],[5,5], 0.27240032], 
                ['blue block', [173, 47],[5,5], 0.26218402]]

obj_ap[3][1] = [70, 136] #'put the yellow block on a blue bowl'
# obj_ap[3][1] = [120, 155] #'put the yellow block between a blue bowl and a green bowl'
# obj_ap[3][1] = [145, 112] #'put the yellow block around a blue bowl'

for obj in obj_bp:
    size = obj[2]
    obj[2] = []
    obj[2].append(obj[1][0]-size[0])
    obj[2].append(obj[1][1]-size[1])
    obj[2].append(obj[1][0]+size[0])
    obj[2].append(obj[1][1]+size[1])
for obj in obj_ap:
    size = obj[2]
    obj[2] = []
    obj[2].append(obj[1][0]-size[0])
    obj[2].append(obj[1][1]-size[1])
    obj[2].append(obj[1][0]+size[0])
    obj[2].append(obj[1][1]+size[1])
print(obj_ap)
# test_prompt = 'A robot arm is trying to manipulate the environment with the language goal: ' + lang_goal+ '.\n'
# test_prompt += 'A detector named VILD is trying to detect the objects in the environment(size [224,224]) from top-up view. \n'
# test_prompt += 'We should deduce the behavior of robot based on those detection result. \n'
# test_prompt += 'before the robot arm picks up the object, VILD detects: \n' 
# for i in range(len(obj_bp)):
#     test_prompt += str(obj_bp[i][0]) +" at postion "+ str(obj_bp[i][1]) + " with size "+str(obj_bp[i][2])+'\n'
# test_prompt += 'after the robot arm place the object, VILD detects: \n'
# for i in range(len(obj_ap)):
#     test_prompt += str(obj_ap[i][0]) +" at postion "+ str(obj_ap[i][1]) + " with  size "+str(obj_ap[i][2])+ '\n'
# test_prompt += 'do you think the robot arm successfully finish the languange goal given the observation? answer only yes or no.if no, you should say reason\n'
# test_prompt += "Here are some rules for you to follow: \n"
# test_prompt += "1.from the given observation, you should reason which objects have been picked and where it s placied (a region).\n"
# test_prompt += "2.you should consider where is the right region to place given the languange goal. \n"
# test_prompt += "3.you should consider the logic of the objects. \n"
# test_prompt += "4.robot arm do not need to interact with the placed objects. \n"
# test_prompt += "5. the region of the object is a rectangle with given width and height centered at position [x,y]. you should consider whether a objects is in the region. \n"
# test_prompt += "1. what is the picked object? \n"
# test_prompt += "2. what is the object should be placed? \n"
test_prompt = 'A robot arm is trying to manipulate the environment with the language goal: ' + lang_goal+ '.\n'
test_prompt += 'A detector named VILD is trying to detect the objects in the environment(size [224,224]) from top-down view. \n'
test_prompt += "the detection result is formulated as rectangle bounding box [x1,y1,x2,y2] where [x1,y1] is the top-left corner and [x2,y2] is the bottom-right corner. \n"
test_prompt += "for example, a bounding box [50,80,70,100] is inside bounding box [40,70,80,110] \n"
test_prompt += "a bounding box [50,80,70,100] is not inside bounding box [40,70,60,80] \n"
test_prompt += 'before the robot arm picks up the object, VILD detects: \n' 
for i in range(len(obj_bp)):
    test_prompt += str(obj_bp[i][0]) +" at position"+ str(obj_bp[i][1])+ " with bounding box region: "+str(obj_bp[i][2]) +'\n'
test_prompt += 'after the robot arm place the object, VILD detects: \n'
for i in range(len(obj_ap)):
    test_prompt += str(obj_ap[i][0]) +" at position"+ str(obj_ap[i][1]) + " with bounding box region: "+str(obj_ap[i][2])+'\n'
test_prompt += 'do you think the robot arm successfully finish the languange goal? answer only reply {Yes,No} \n'


# model = "gpt-3.5-turbo-0613"
model = "gpt-3.5-turbo"
print(test_prompt)
completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI.\n Knowledge cutoff: 2021-09 \n Current date: 2023-06-29 \n"},
        {"role": "user", "content": test_prompt}
    ],
    temperature=0.,

    )
print(completion)
