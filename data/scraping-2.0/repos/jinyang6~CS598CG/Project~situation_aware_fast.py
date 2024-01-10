from openai import OpenAI
import json
import ast
import time


def create_gpt_client():
    client = OpenAI(api_key = "*************************")
    return client


def get_gpt_response(client, content, prev_messages, model):
    prev_messages.append({"role": "user", "content": content}) 
    chat = client.chat.completions.create(model=model, messages=prev_messages)
    response = chat.choices[0].message.content
    prev_messages.append({"role": "assistant", "content": response})
    return response, prev_messages


def read_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data
    

def write_json(file_name, json_object):
    with open(file_name, 'w') as f:
        json.dump(json_object, f, indent=4)

    
def build_situation_aware(model, client, IoT_device_location, room_setup):
    """
        IoT_device_location: IoT_device_location.json
        room_setup: room_setup.json
    """
    messages = [ {"role": "system", "content":  "You oversees IoT devices."} ] 
    gpt_request =   """
                    User is authorized to control everything
                    IoT_device_location: %s
                    room_setup: %s
                    NO NEED FOR RESPONSE
                    """
    request_string = gpt_request % (IoT_device_location, room_setup)
    _, prev_messages = get_gpt_response(client, request_string, messages, model)
    return prev_messages

def is_benign(model, client, prev_messages, user_location, user_command=None, silent_triggered_actions=None, actual_triggered_actions=None):
    
    """
        user_location: location of user
        user_command: user's direct command, eg: voice command, app command
        silent_triggered_actions: silent commands, eg: motion sensor, open up the magnetic_induction_cooktop
        actual_triggered_actions: actual triggered actions, eg: actual packet, overprivileged accesses, event spoofing
        
        Returns: True when all actions are correct, list of other possible actions, prev_messages
                 False when overprivileged accesses or event spoofing, list of possible reasoning of why actions are considered as attack, prev_messages 
    """    
    gpt_request =   """
                    User_command is authorized to control everything
                    user_location: %s # current location of user
                    user_command: %s # user's direct command
                    silent_triggered_actions: %s # user setted silent commands, eg: motion sensor, timed commands, this is CORRECT actions
                    actual_triggered_actions: %s # actions that just going to be triggered, do not know if correct or not
                    
                    Give me the result of following, only the outputs, as IoT device owner/user, should following common human behaviors, user in same location should trigger related actions with related devices&actions, and should not trigger unrelated actions with unrelated devices, reason with locality, common human behaviors, and common sense, user directly triggered are all correct
                    only the output and no additional explanations or reasoning or text, strictly following the format
                    If overprivileged_accesses or event_spoofing doesn't occure:
                        return True 
                    else:
                        # when overprivileged accesses or event spoofing occurs, illogical triggering, not common human behaviors, not common sense                
                        return False
                    """
    request_string = gpt_request % (user_location, user_command, silent_triggered_actions, actual_triggered_actions)
    response, prev_messages = get_gpt_response(client, request_string, prev_messages, model)
    is_correct = response[:4] == "True"
    yield is_correct, prev_messages
    gpt_request = """
                    To above situation,
                    User_command is authorized to control everything
                    user_location: as above # current location of user
                    user_command: as above # user's direct command
                    silent_triggered_actions: as above # user setted silent commands, eg: motion sensor, timed commands, this is CORRECT actions
                    actual_triggered_actions: as above # actions that just going to be triggered, may be overprivileged_accesses or event_spoofing!
                    Give me the result of following, only the outputs, as IoT device owner/user, should following common human behaviors, user in same location should trigger related actions with related devices&actions, and should not trigger unrelated actions with unrelated devices, reason with locality, common human behaviors, and common sense, user directly triggered are all correct
                    only the output and no additional explanations or reasoning or text, strictly following the format
                    If previouse response doesn't have overprivileged_accesses or event_spoofing:
                        # if previouse response is True
                        # other actions than already triggered actions should be considered as well
                        list of other possible actions in the format of [ {"location": "LOCATION", "device": "DEVICE", "action": "ACTION"}, ...], strcit minimal actions, do not trigger any additional unrelated actions, must be related to user_command or related devices, close distance devices can be triggered, ast.literal_eval parsable
                    else:
                        # if previouse response is False
                        # when overprivileged accesses or event spoofing occurs, illogical triggering, not common human behaviors, not common sense                
                        list of possible reasoning of why actions are considered as overprivileged_accesses or event_spoofing eg: ["Following are overprivileged_accesses 1. ...", ...], ast.literal_eval parsable
                    """
    response, prev_messages = get_gpt_response(client, gpt_request, prev_messages, model)
    content = ast.literal_eval(response)
    yield content, prev_messages





# gpt-3.5-turbo
# gpt-4
# gpt-4-turbo
print("Wait...")
model = "gpt-4"
client = create_gpt_client()
IoT_device_location = read_json("IoT_device_location.json")
room_setup = read_json("room_setup.json")
prev_messages = build_situation_aware(model, client, IoT_device_location, room_setup)    

# time it
start_time = None
end_time = None

while input("Stop?") != "y":
    user_location = input("user_location: ")
    user_command = input("user_command: ")
    silent_triggered_actions = input("silent_triggered_actions: ")
    actual_triggered_actions = input("actual_triggered_actions: ")
    
    start_time = time.time()
    for output in is_benign(model, client, prev_messages, 
                                                    user_location=user_location,
                                                    user_command= None if user_command == "" else user_command, 
                                                    silent_triggered_actions= None if silent_triggered_actions == "" else silent_triggered_actions, 
                                                    actual_triggered_actions= None if actual_triggered_actions == "" else actual_triggered_actions,
                                                    ):
        result, prev_messages = output
        end_time = time.time()
        
        if type(result) == bool:
            is_correct = result
            print("is_correct: ", is_correct)
            print("Time taken: ", end_time - start_time)
            start_time = time.time()
        else:
            content = result
            print("content: ", content)
            print("Time taken: ", end_time - start_time)
    
write_json("prev_messages.json", prev_messages)

