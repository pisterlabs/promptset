import os
import openai
import json
import requests
import random

def generate_response(prompt, model='turbo'):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=1,
        max_tokens=400,
    )
    if model=='davinci':
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=1,
            max_tokens=400,
        )
    return response

openai.api_key = #TODO
mutual = 0 #0 for D2I and 1 for I2D
ins = 0 #0 for zero-shot, 1 for 3-shot and 2 for 3-shot*
model = 'turbo'
fw = open('log/test_turbo_D2I.txt', 'a')

if mutual:
    lines = open('data/test_out.txt').readlines()
    prps = open('data/similar_instance_out.txt').readlines()
else:
    lines = open('data/test_inp.txt').readlines()
    prps = open('data/similar_instance.txt').readlines()


inps = open('data/train_inp.txt').readlines()
outs = open('data/train_out.txt').readlines()

def getrandom(N, mutual=0):
    rec = ''
    for i in range(N):
        rd = random.randint(0, len(inps)-1)
        if mutual==0:
            rec+=' DESCRIPTION: '+inps[rd].strip('\n')+' INSTRUCTION: '+outs[rd].strip('\n')
        else:
            rec+=' INSTRUCTION: '+outs[rd].strip('\n')+' DESCRIPTION: '+inps[rd].strip('\n')
    return rec

for ind in range(790):
    line= lines[ind]
    if ins==2:
        prp = prps[ind*3]+prps[ind*3+1]+prps[ind*3+2]
        if len(prp.split(' '))>1650:
            prp = prps[ind*3]+prps[ind*3+1]
    elif ins==1:
        prp = getrandom(3, mutual)
        while len(prp.split(' '))>1650:
            prp = getrandom(3)
    if mutual:
        if ins>0:
            prp1= 'You are now a synthetic literature writing assistant. Generate the description according to the given instructions as shown in the following instances. '+prp+' Now the INSTRUCTION is: '+line.strip('\n')+' In any case, please generate the DESCRIPTION. '
        else:
            prp1 = 'You are now a synthetic literature writing assistant. Generate the synthetic description according to the given instructions. INSTRUCTION: '+line.strip('\n')+' In any case, please generate the DESCRIPTION. '
    else:
        if ins>0:
            prp1= 'You are now a synthetic experiment assistant. Generate the synthetic instructions according to the given descriptions as shown in the following instances. '+prp+' Now the DESCRIPTION is: '+line.strip('\n')+' In any case, please generate the INSTRUCTION. '
        else:
            prp1 = 'You are now a synthetic experiment assistant. Generate the synthetic instructions according to the given descriptions. Use "[  ]", "&" and ":" to mark operations, split different segments and split the name and value of parameters, such as "[ OPERATION ] PARAMETER1 NAME: VALUE1 & PARAMETER2 NAME: VALUE2 &". Operations include add, wash, filter, dry, extract, recrystallize, quench, partition, transfer, yield, distill, evaporate, column, settemp and reflux. Parameters include time, temperature, phase, reagent, mass, composition, speed, mole, batch, volume, comcentration and note. Now the DESCRIPTION is: '+line.strip('\n')+' What is the INSTRUCTION? '
    
    tmp = {}
    tmp['role'] = 'user'
    tmp['content'] = prp1

    response = generate_response([tmp])
    print(ind, response['choices'][0]['message']['content'].strip('\n'))
    
    fw.write(response['choices'][0]['message']['content'].replace('\n', ' ')+'\n')
    fw.flush()
    
fw.close()
