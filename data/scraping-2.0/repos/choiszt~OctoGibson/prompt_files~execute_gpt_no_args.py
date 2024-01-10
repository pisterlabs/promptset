import os
import json
import yaml

# import omnigibson as og

# from robot_action import *
import parse_json
import query_new as query
from imp import reload
import env_utils_gpt as eu 
import openai
# from gpt_request import gpt_request
from gpt_request_azure import gpt_request
import argparse


def parse_args():
    description = "EVLM_gpt_process"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-save", "--save_path", type=str, help="data save path", required=True)
    return parser.parse_args()


def gpt_process(args):
    try:
        task_we_have=os.listdir("/Users/liushuai/Desktop/omnigibson-master/prompt_files/data")
        with open("/Users/liushuai/Desktop/omnigibson-master/EVLM_Task/task_0908.json" ,"r")as f:
            task_json=json.load(f)
        for task in sorted(list(task_json.keys())):
            if task not in task_we_have:
                now_task=task
        
        # sorted_task_list=sorted(list(task_json))

        #the implementation of args
        # for ele in task_json.keys():
        #     if ele not in loaded_task:
        #         loaded_task.append(ele) #(a,b)->(a,b,c)
        #         now_task=task_json[ele]
        #         break

        
        task_path=f"./prompt_files/data/{now_task}"
        save_path = eu.f_mkdir(task_path)
        # action_path = os.path.join(save_path, 'action.py')


        save_path = args.save_path
        # main task loop
        
        main_task_flag = False
        subtask_iter = 1
        gpt_query = query.Query()
        for i in range(1, 15):
            eu.f_mkdir(os.path.join(save_path, f"subtask_{i}"))

        while True:

            # make the directory
            sub_save_path = eu.f_mkdir(os.path.join(save_path, f"subtask_{subtask_iter}"))
            
            # init pipeline for each subtask
            while True:
                if os.path.exists(os.path.join(sub_save_path, 'task1.json')): #TODO align with "task1"
                    break   
            while True:
                statinfo = os.stat(os.path.join(sub_save_path, 'task1.json')) 
                if statinfo.st_size > 0:
                    break
            human_info = parse_json.parse_json(path=os.path.join(sub_save_path, "task1.json"))
            
            
            # subtask loop, when a subtask is finished, close the loop
            while True:
                system_message = gpt_query.render_system_message()
                human_message = gpt_query.render_human_message(
                    scene_graph=human_info[0], object=human_info[1],
                    inventory=human_info[2], task=human_info[3] 
                )
                all_messages = [system_message, human_message]
                content=system_message.content+"\n\n"+human_message.content
                eu.save_input(sub_save_path, human_message.content)
                print("start query")
                response=gpt_request(content)
                try:
                    answer = gpt_query.process_ai_message(response)
                except Exception as e:
                    answer = str(e)
                    print(answer)
                eu.save_response(sub_save_path, answer)
                
                while True:
                    if os.path.exists(os.path.join(sub_save_path, 'feedback.json')):
                        break
                while True:
                    feedbackinfo = os.stat(os.path.join(sub_save_path, 'feedback.json')) 
                    if feedbackinfo.st_size > 0:
                        break

                with open(os.path.join(sub_save_path, 'feedback.json')) as f:
                    data = json.load(f) 
                
                main_task_flag = data['main_succeed']      
                if data['critic'] == 'succeed':
                    print('Task succeed!')
                    gpt_query.record_history(subtask=data['subtask'], code=data['code'], error=data['error'])
                    break
                else:
                    if data['reset']:
                        gpt_query.record_history(subtask=answer['subtask'], code=answer['code'], error=data['error'])
                        break
                    else:
                        gpt_query.record_history(subtask=answer['subtask'], code=answer['code'], error=data['error'])
                        break
            

            
            # reset parameters
            subtask_iter += 1
            
            if subtask_iter>15:
                print(f"already attempt {subtask_iter} time, it is too long!")
                break

            if main_task_flag:
                break
        return 0
    except:
        raise Exception

if __name__ == "__main__":
    args = parse_args()
    gpt_process(args)