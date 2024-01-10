import openai
import json




# Read and write the global time between two forms: 07:50  - 470 min from the midnight
def time(s):
    return 600*int(s[0])+60*int(s[1])+10*int(s[3])+int(s[4])

def inv_time(s):
    return str(s//600) + str((s%600)//60) + ":" + str((s%60)//10) + str(s%10)


###
#  Take plan into a part of prompt as a base of inference for importance.
#  Parameter: path of agent's plan and the game's time;  Output: two strings as plan done and plan to do
def get_plan(file_path,global_time):
    with open(file_path, 'r') as planj:
        plan = json.loads(planj.read())
    text1 = ""
    text2 = "Now it's {}. The plan for the rest of day is : \n".format(inv_time(global_time))
    ntask = len(plan)
    for i in range(ntask):
        if len(plan["task_{}".format(i+1)]) == 3 :
            text2 += "From " + plan["task_{}".format(i+1)][0] + " to " + plan["task_{}".format(i+1)][1] + ", " + plan["task_{}".format(i+1)][2] + ". \n"
        else:
            sub = plan["task_{}".format(i+1)][3]
            nsub = len(sub)
            for j in range(nsub):
                ssub = sub["sub_task_{}".format(j+1)]
                text1 += ssub[2] + "; \n"
    return text1,text2

# A sentence to restrict the answer
ex = """The response should be given in the form of only an array of numbers with two significant figures(Example: [0.20,0.48]).Don't answer anything else."""


# Version 1 of prompt generation
def p1_ger(name,file_path,event,global_time):
    plan1,plan2 = get_plan(file_path,global_time)
    m = "Based on what have already been done today by " + name + " (between &&, in chronological order from morning) : && \n" + plan1 + "&& \n and his/her plan (between ##): ## \n"   + plan2 + "## \n, try to determine the importance coefficient to him/her (normalized in [0,1], 1 means that the event is extremely important, 0 means that the event is not important at all) of these events(separated by ;): \n" + event + "\n"
    m += """Here is an possible example: Bob's plan is to prepare the maths exam tomorrow , then the importance of event "Review the maths exercises" is 0.9 while that of event "Mow the lawn of garden" is 0.1.\n"""
    m += ex
    return m


# Version 2 of prompt generation: take identity and emotion as the second base of inference for importance, and combine it with plans
# Parameter: name of agent, id of agent, target event, game time;  Output: the final prompt
def p2_ger(agent,id,event,global_time):
    with open(f"back_end/memory/{id}/identity.json", 'r') as idenj:
        identity =json.loads(idenj.read()) 
    plan1,plan2 = get_plan("back_end/memory/{id}/plan.json", global_time)    
    emotion = agent.emotion.print_emotions()
    #identity = "name: Jon; age: 34; personnality: friendly, hardworking, respectful, and kind; lifestyle: Jon goes to bed around 11pm, awakes up around 7am. profile: Jon is an enginner at Orange. He loves to exercice and he is handsome. "
    #emotion = """ Here is its current emotional state: (0 means the emotion is not felt, 1 means it is felt to the maximum) -             
            # Joy at 0.7,
            # Sadness at 0.1,
            # Fear at 0.1,
            # Love at 0.2,
            # Hate at 0.1,
            # Pride at 0.4,
            # Shame at 0.1 """
    m = "Known the person's identity: \n" + identity + "\n, and his emotions: \n" + emotion #+ "and his personality: " + personality
    m += """\nAnd based on what he/she has already done today (between &&, in chronological order from morning) : && \n""" + plan1 + "&& \n and his plan (between ##): ## \n"   + plan2 + """## \n, try to determine the importance coefficient to him (normalized in [0,1], 1 means that the event is extremely important, 0 means that the event is not important at all) of these events(separated by ";" and included in "{}"): """ + event + "\n"   
    m += ex
    return m



def get_completion(prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0):
    print("get completion")
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, #the randomness of the output
        max_tokens=max_tokens,
    )
    print("get completion done")
    return response['choices'][0]['message']['content']




  

