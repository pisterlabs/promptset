########## GPT setting code ##########
import openai
import pdb
import json
from tqdm import tqdm
import argparse
import math
import time

def main(nuscenes_dir: str, json_dir: str):
    My_OpenAI_key = ''
    openai.api_key = My_OpenAI_key

    chat_completion = openai.ChatCompletion()
    temperature = 0.5
    max_tokens = 2000

    # for merged caption with location data in one image
    # ask_zero = """I am now working on the vehicle prediction of self-driving cars.
    #             To increase performance, we use ego central camera view images to generate text and try to use that text.
    #             The pre-trained model was fine-tuned on a custom dataset and used that model to create a capture of objects corresponding to 2d Bboxes was created.
    #             It also created future movements through data processing.
    #             You are a machine box that fuses some datas well and make it into a single text, if I provide the generated caption, GT maneuver, and GT class of each object.
    #             In particular, I will provide data on multiple objects that exist in one image.
    #             So please understand the relationship between the data that exist in one image and create a caption for each object.
    #             Here are some rules for you.
    #             Rule 1: Don't mention GT class but check if it fits well with the original caption and create a caption to follow GT class if it's wrong.
    #             Rule 2: When creating fused captions, make them without the object's location text like 'left', 'right', 'left to right', 'right to left' that exists in the provided original captions.
                # Rule 2: Remove any location-specific text from the caption like 'left side of the road', 'in right lane', 'from left to right', 'from right to left'.
    #             Rule 3: Create captions to match the number of objects I provided. If you provided three object, you should create three captions. If you provided four object, you should create four captions.
    #             Rule 4: Don't make merged caption about whole image.
                # Rule 1: Check the caption against the GT class. If the caption doesn't match the GT class, adjust it without mentioning the GT class explicitly.
    #             I'll give you two output examples that is expected from you, each of which is about an object and corresponding caption that exist in one image have two objects.
    #             object 1: The white SUV is parked and is anticipated to stay stationary.
    #             object 2: A construction worker is expected to remain standing still.
    #             I will give you list of brackets, each bracket contains about one object.
    #             Please format the result in the order of provided."""

    # for caption for each object
    # ask_zero = """I am now working on the vehicle prediction of self-driving cars.
    #             To increase performance, we use ego central camera view images to generate text and try to use that text.
    #             The pre-trained model was fine-tuned on a custom dataset and used that model to create a capture of objects corresponding to 2d Bboxes was created.
    #             It also created future movements that called maneuver through data processing.
    #             You are a machine box that fuses some datas well and make it into a single text, if I provide the generated caption, GT maneuver, and GT class of an object.
    #             I'll provide you with the rules you need to follow when you create caption.
    #             Rule 1: Remove any location-specific text from the caption like 'left side of the road', 'in right lane', 'from left to right', 'from right to left', 'away from the ego car'.
    #             Rule 2: Incorporate the GT maneuver into the caption to indicate expected future movement.
    #             I'll give you an input example for you, and output example that is expected from you.
    #             input:
    #             (GT class: human.pedestrian.adult
    #             caption: a pedestrian wearing a white shirt and black pants, walking on the right side of the road, away from the ego car
    #             GT maneuver: right_turn)
    #             expected output:
    #             fused caption: a pedestrian wearing a white shirt and black pants, and is anticipated to turn right."""

    ask_zero = """
    Your Role: You are a writer tasked with generating descriptive captions about objects without including their location information.

    Inputs Explained:
    1. Caption: Describes an object but might contain location info which we don't want.
    2. GT class: The actual type of the object.
    3. Maneuver: Predicted future movement of the object.

    Your Task:
    - Create two distinct versions of a single caption.
    - DO NOT include location information like 'left side', 'right lane', 'away from the ego car', 'in the ego lane' etc.
    - If the object described in the Caption is different from the GT class, craft your caption using only the GT class and Maneuver.
    - If the action described in the Caption does not align with the Maneuver, adjust the description to fit the provided Maneuver.
    - Explicitly mention the object's expected maneuver using the provided "Maneuver" input.

    Input format example:
    Caption: a white suv driving in the left lane, away from the intersection, in the rain
    GT class: vehicle.car
    Maneuver: straight

    Output format: 
    caption1: ~~~
    caption2: ~~~
    """
    # print(ask_zero)
    
    messages = [{'role': "user", "content": ask_zero}]
    response = chat_completion.create(
                            messages=messages, 
                            model='gpt-3.5-turbo',
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
    answer = response.choices[0].message['content'].strip()
    print(answer)
    # answer
    # messages.append({'role': "assistant", "content":answer})
    # # request
    # messages.append({'role': "user", "content":"""(GT class: vehicle.car
    #                                                 caption: a white sedan parked on the left side of the intersection
    #                                                 GT maneuver: right_turn)"""})
    # messages.append({'role': "assistant", "content": """A white sedan is parked on the intersection and is anticipated to make a right turn."""})
    # # request
    # messages.append({'role': "user", "content":"""(GT class: vehicle.car
    #                                                 caption: a white suv driving in the left lane, away from the intersection, in the rain
    #                                                 GT maneuver: straight)"""})
    # messages.append({'role': "assistant", "content": """A white SUV is driving, and is expected to continue straight, in the rain."""})

    start_time = time.time()
    with open(json_dir, "r") as f:
        data = json.load(f)
    
    new_data = {}
    image_count = 0
    for image_path, object_list in tqdm(data.items(), desc="Processing images", position=0):
        new_obj_list = []
        for obj in tqdm(object_list, desc=f"Processing objects in an image", position=1, leave=False):
            if "chatgpt_caption" in obj and "chatgpt_caption_v2" not in obj:
                GT_class = obj["category_name"]
                GT_maneuver = obj["maneuver"]
                caption = obj["caption"]
                prompt = f"""(Caption: {caption}\nGT class: {GT_class}\nmaneuver: {GT_maneuver}\n)\n"""
            
                messages.append({'role': "assistant", "content": prompt})
                max_retries = 10  # 최대 재시도 횟수
                for _ in range(max_retries):
                    try:
                        response = chat_completion.create(
                                            messages=messages, 
                                            model='gpt-3.5-turbo',
                                            max_tokens=max_tokens,    
                                            temperature=temperature,
                                            )
                        answer = response.choices[0].message['content'].strip()
                        obj["chatgpt_caption_v2"] = answer
                        break  # 요청이 성공하면 재시도 루프를 빠져나옵니다.
                    except openai.error.ServiceUnavailableError:
                        print("Service is unavailable. Retrying...")
                        time.sleep(5)  # 잠시 대기하고 다시 시도합니다.
                    except Exception as e:
                        print("An unexpected error occurred:", str(e))
                        time.sleep(10) # 다른 예외가 발생하면 다시 시도

                messages.pop()
                time.sleep(0.1)

            new_obj_list.append(obj) 
        
        new_data[image_path] = new_obj_list
        # image_count += 1
        # if image_count != 0 and image_count % 10000 == 0:
        #     with open(f"/workspace/nuscenes_bp/data/nuscenes/v1.0-trainval/with_text_v2.0/v1_rest2_thr_{image_count}.json", "w") as f:
        #         json.dump(new_data, f, indent=4)
        #     new_data = {}
            
    if new_data:
        # new_json_dir = "/workspace/nuScenes/v1.0-trainval/image_annotations_with_caption_maneuver.json"
        # new_json_dir = "/workspace/blip2_mod/docs/_static/nuscenes_test_few.json"
        with open(f"/workspace/nuscenes_bp/data/nuscenes/v1.0-trainval/with_text_v2.0/final_v2.json", "w") as f:
            json.dump(new_data, f, indent=4)
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuscenes_dir', default="/workspace/nuscenes_bp/data/nuscenes/")
    # parser.add_argument('--json_dir', default="/workspace/nuscenes/v1.0-trainval/rest_professor.json")
    parser.add_argument('--json_dir', default="/workspace/nuscenes_bp/data/nuscenes/v1.0-trainval/with_text_v2.0/final_merged_v2.json")
    args = parser.parse_args()
    exit(main(args.nuscenes_dir, args.json_dir))