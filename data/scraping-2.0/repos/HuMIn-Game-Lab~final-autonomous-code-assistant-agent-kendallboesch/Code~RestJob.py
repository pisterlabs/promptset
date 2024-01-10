import os 

import openai
import sys
import json 
import seaborn as sns


# instruction = 'errorsolve'
# instruction = 'flowgen'
# openai.api_base="http://localhost:4891/v1"
# promptFile = 'errors.json'
# promptFile = 'Data/LLMFlowscriptPrompt.txt'
instruction = sys.argv[1]
openai.api_base=sys.argv[2]
promptFile = sys.argv[3]
model = "NA"
openai.api_key = "not needed for a local LLM"

# print(f'instruction: {instruction}')

if instruction == 'errorsolve':

    file = open(promptFile)
    data =json.load(file)

    inputFormat = "{\n{\'colNum\' : #}, \n {\'errorMessage\': \'\'},\n{\'file\': \'\'},\n{\'lineNum\': #},\n{\'nextLine\':\'\'},\n{\'previousLine\':\'\'},\n{\'resDescr\': \'\'},\n{\'src\': \'\'},\n{\'srcResolved\': \'\'}\n}\n"
    erRes = "{\"colNum\" : #, \n \"errorMessage\": \"\",\n\"file\": \"\",\n\"lineNum\": #,\n\"nextLine\":\"\",\n\"previousLine\":\"\",\n\"resDescr\": \"\",\n\"src\": \"\",\n\"srcResolved\": \"\"}\n"
    basePrompt=f"This is the format of an error object:\n {inputFormat}\n"
    instruction="For each error object given, provide the resolved C++ code in the \'srcResolved\' member, and provide a description of the fix in the \'resDescr\' member.\n Leave the incorrect code in the 'src' member.\n"
    details="\'srcResolved\' should only contain valid c++ code. Replace \"file.cpp\" with the \'file\' memeber in the error object.\n "
    noPrompt = f"\nDo not prompt your response.\n"
    responseFormatOpen = "{\n\"file.cpp\": [\n"
    responseFormatClose = "\n]\n}\n"
    responseformat = f"Your response should be in the following valid json format:\n{responseFormatOpen}{erRes},{erRes}{responseFormatClose}"
    file.close()

    #print(prompt)
    if os.path.exists("Data/convo.json"):
        with open("Data/convo.json", 'r') as json_file:
            write = json.load(json_file)
    else: 
        write = []
    
    # appending = False
    # if os.path.exists('errorhistory.json'):
    #   with open('errorhistory.json', 'r') as outFile:
    #     writer = json.load(outFile)
    #     appending = True
    # else:
    #     writer = []

    for file, errors in data.items():
        # print(f"File: {file}")
        # prompt = basePrompt + instruction + details + responseformat + noPrompt
        prompt = basePrompt + responseformat + instruction + details + noPrompt
        for error in errors:
            prompt = prompt + str(error)
            # print(f"Error: {error}")
        


        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=750,
            temperature=0.28,
            top_p=0.95,
            n=1,
            echo=True,
            stream=False
        )
        
        text = response['choices'][0]['text']
        LLMresponse = text[len(prompt):]
        
        lowerLLMres = LLMresponse.lower()
        if lowerLLMres.find('```json\n') != -1:
            print('had ```json\n\n')
            LLMresponse = LLMresponse[8:]
            LLMresponse = LLMresponse[:-3]

        # LLMResAppend = LLMresponse
        
        # if appending:
        #     LLMResAppend = LLMresponse[1:-1]
            
        # js = json.loads(LLMResAppend)
        # writer.append(js)
        
        write.append(response)
    with open("Data/convo.json", 'w') as json_file:
        json.dump(write, json_file, indent=4)
    fixWriter = []
    fixWriter.append(LLMresponse)
    with open('fixedErrors.json', 'w') as out:
        out.write('\n'.join(map(str,fixWriter)))
        # json.dump(fixWriter, out, indent=4)
        # json.dump(writer, outFile, indent=4)
    # with open('errorhistory.json', 'w') as outFile:
    #     outFile.write('\n'.join(map(str,writer)))
        
    print('fixedErrors.json')
    
elif instruction == 'flowgen':
    basePrompt=''
    with open('Data/LLMFlowScriptSingleShot.txt','r') as file:
        basePrompt = file.read(); 
    
    with open(promptFile) as file:
        prompt = file.read()
          
    basePrompt = basePrompt + prompt + '\nDo not include anything other than the Flowscript script in your response\n'
        
        
    response = openai.Completion.create(
        model=model,
        prompt=basePrompt,
        max_tokens=750,
        temperature=0.28,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )

    text = response['choices'][0]['text']
    LLMresponse = text[len(basePrompt):]
    with open('Code/Flowscript/script.md', 'w') as f:
        f.write(LLMresponse)
    print('Code/Flowscript/script.md')
    