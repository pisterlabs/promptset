#!/usr/bin/env python3

import openai
import json
import os
import rospy
from turtle_gpt.srv import GPTPrompt, GPTPromptResponse

openai.api_key = os.getenv('OPENAI_API_KEY')
# openai.organization = os.getenv('OPENAI_ORG')

class PromptGenerator:
    def __init__(self, desc_folder_path):
        self.desc_folder_path = desc_folder_path
        self.__index_file = os.path.join(self.desc_folder_path, 'index.json')
        self.__examples_dir = os.path.join(self.desc_folder_path, 'examples')

    def __parseJSON(self, filePath):
        path = os.path.join(filePath)
        with open(path, 'r') as json_file:
            data = json_file.read()
            jsonData = json.loads(data)
        return jsonData
    
    def __getParamDescStringForMethod(self, param):
        paramtype = param.get('type')
        desc = param.get('description')
        opts = param.get('options')
        valrange = param.get('range')
        res = f'It should have a {paramtype} type value. '
        if valrange:
            start = valrange.get('start')
            end = valrange.get('end')
            res += f"The value ranges from {start} to {end}. "
        elif opts:
            default = param.get('default')
            res += f'This will take values from the following {", ".join(opts[:-1])} and {opts[-1]}. Use {default} as default values if not mentioned or cannot decide. '
        res += f'Additional informaton: {desc}'
        return res

    def __getMethodDescString(self, method):
        name = method['name']
        description = method['description']
        params = method['params']

        res = f'• {name} - {description}, it has the following parameters :\n'
        for param in params:
            # val = params[param]
            val = param
            res += "\t\t\t" + param['name'] + ": " + self.__getParamDescStringForMethod(val) + '\n'
        return res
    
    def __generateExamples(self, examplesArr):
        res = 'Examples: The following are the prompts and the expected output:\n'
        for eg in examplesArr:
            res += f"prompt: {eg['prompt']}"
            res += f"\noutput: {eg['returns']}"
        return res
        
    def generateBasePrompt(self):
        desc = self.__parseJSON(self.__index_file)
        examples = [self.__parseJSON(os.path.join(self.__examples_dir, file)) for file in os.listdir(self.__examples_dir)]
        examplesString = self.__generateExamples(examples)

        methodsDesc = '\n'.join([self.__getMethodDescString(method) for method in desc['api']])

        prompt = f'''   Consider the following robot named - {desc['name']}.
                        Description - {desc['description']}
                        You can interact with the robot using the following high-level operations.

                        {methodsDesc}

                        Note: All parameters to the high level operations described are requeired and should only be in given range/values/options. If can't decide, use default option
                        Note: Keep the 'params' attribute as array, and all parameters in the array should be in correct order
                        
                        You will be given human language prompts, and you need to return an array of operations that are JSON conformant to the ontology. Any action that is not in the ontology should be ignored.
                        Each operation you produce should be a valid JSON structure as described above.

                        {examplesString}

                        Say Yes if you are ready.
                  '''
        print('prompt', prompt)
        return prompt
    

class GPTPrompterService:
    def __init__(self):
        rospy.init_node('GPTPromptGenerator')
        srv = rospy.Service('GPTPrompt', GPTPrompt, self.handler)
        self.promptHistory = [{"role": "system", "content": "You are a helpful assistant."}]
        self.__callBasePrompt()
        

    def __callBasePrompt(self):
        try:
            basePrompt = PromptGenerator(os.getenv('DESCRIPTION_FOLDER_PATH')).generateBasePrompt()
            self.promptHistory.append({"role": "user", "content": basePrompt})
            response = self.__askGPT(basePrompt)
            self.promptHistory.append({"role": "system", "content": response})
        except Exception as e:
            print(f"An exception occured in Base Prompt Generation", e)
    
    def __pickJSONFromResponseContent(self, content):
        i = content.find('[')
        j = content.rfind(']') + 1
        json_content = content[i:j]
        print(json_content)
        return json_content

    def __askGPT(self, prompt):
        self.promptHistory.extend([{"role": "user", "content": prompt}])
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.promptHistory,
                temperature=0
            )
        except openai.InvalidRequestError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
        content = response.choices[0].message['content'].strip()
        self.promptHistory.extend([{"role": "assistant", "content": content}])
        return content

    def handler(self, data):
        prompt = data.prompt
        finalPrompt = prompt

        res = GPTPromptResponse()
        res.response = self.__pickJSONFromResponseContent(self.__askGPT(finalPrompt))
        print(res.response)

        return res

def main():
    GPTPrompterService()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
