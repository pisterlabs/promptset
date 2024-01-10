import os
import json
import yaml
import time
# import omnigibson as og

# from robot_action import *
import parse_json
import query_new as query
from imp import reload
import env_utils_gpt as eu 
import openai
# from gpt_request import gpt_request
from gpt_request_azure import gpt_request

def gpt_process(save_path, openai_api_key):
    # main task loop
    
    main_task_flag = False
    subtask_iter = 1
    gpt_query = query.Query(openai_api_key=openai_api_key)
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
            success=True
            while success:
                try:
                    response=gpt_request(content)
                    success=False
                except Exception as e:
                    print(f'Error:{e}')
                    if "have exceeded" in str(e):
                        print("sleeping for 3 seconds")
                        success=True
                        time.sleep(3)
                    else:
                        success=False
                        response={"error_message":str(e)}
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
        
        if subtask_iter>14:
            print(f"already attempt {subtask_iter} time, it is too long!")
            break
                
        if main_task_flag:
            break

api_key="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"
with open("/shared/liushuai/OmniGibson/prompt_files/919TODO/failed_todo.json","r")as f:
    task=json.load(f)
i=8
EVLM_name=sorted(list(task))[i]
task_name=task[EVLM_name]['task_name']
gpt_name=task[EVLM_name]['detailed_name']
scene=task[EVLM_name]['env']
print(i)
print(f"EVLM:{EVLM_name}")
print(f"task_name:{task_name}")
print(f"gpt_name:{gpt_name}")
gpt_process(save_path=f"/shared/liushuai/OmniGibson/prompt_files/data/{gpt_name}",openai_api_key=api_key)