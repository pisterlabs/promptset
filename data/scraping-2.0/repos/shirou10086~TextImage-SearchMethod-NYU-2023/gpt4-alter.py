import openai
import time
import os
import numpy as np
import pandas as pd
import re  # 导入正则表达式模块

subfolderList =['Adrian', 'Anaheim', 'Andover', 'Angiola', 'Annawan','Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Cantwell', 'Capistrano', 'Colebrook', 'Cooperstown', 'Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Goffs', 'Greigsville', 'Hainesburg', 'Hambleton', 'Haxtun', 'Hillsdale', 'Hometown', 'Hominy', 'Kerrtown', 'Maryhill', 'Mesic', 'Micanopy', 'Mifflintown', 'Mobridge', 'Monson', 'Mosinee', 'Mosquito', 'Nemacolin', 'Nicut', 'Nimmons', 'Nuevo', 'Oyens', 'Pablo', 'Parole', 'Pettigrew', 'Placida', 'Pleasant', 'Quantico', 'Rancocas', 'Reyno', 'Ribera', 'Roane', 'Roeville', 'Rosser', 'Roxboro', 'Sanctuary', 'Sands', 'Sawpit', 'Scioto', 'Seward', 'Shelbiana', 'Silas', 'Sisters', 'Sodaville', 'Soldier', 'Spencerville', 'Spotswood', 'Springhill', 'Stanleyville', 'Stilwell', 'Stokes', 'Sumas', 'Superior', 'Swormville']

#folderList=['Denmark', 'Dryville', 'Dunmor', 'Eagerville', 'Eastville', 'Edgemere', 'Elmira', 'Eudora', 'Greigsville', 'Hambleton', 'Haxtun', 'Hillsdale', 'Hometown', 'Hominy', 'Maryhill', 'Mesic', 'Mifflintown', 'Mobridge', 'Monson']
folderList=['Adrian','Andover','Angiola','Ballou','Beach','Elmira']


notlist=['Anaheim','Azusa','Ballou', 'Goffs', 'Hainesburg', 'Kerrtown', 'Micanopy', 'Mosquito', 'Nemacolin', 'Nicut', 'Nimmons', 'Pettigrew', 'Placida', 'Reyno', 'Roeville', 'Sanctuary', 'Scioto', 'Shelbiana', 'Silas', 'Soldier', 'Spencerville', 'Spotswood', 'Springhill', 'Stilwell', 'Stokes', 'Superior', 'Swormville' ]
folderList = [folder for folder in folderList if folder not in notlist]

chars_sent_this_minute = 0
current_minute = time.time() // 60

# Replace 'your-api-key-here' with your actual OpenAI API key
openai.api_key = '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'#jinli
#openai.api_key = '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'
def summarize_with_gpt4(subdir_path):
    # Define a prompt for GPT-4 to generate a summarized layout description
    combined_text = ""

    # Collect descriptions of all images in the scene
    for filename in os.listdir(subdir_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(subdir_path, filename)
            with open(file_path, 'r') as file:
                combined_text += file.read() + '\n'

    prompt = "Summarize the following room descriptions into a 300-word layout: " + combined_text[:4000]

    try:
        prompt_message = "Given the combined text from the images, provide a detailed description focusing on the spatial relations, such as positions of furniture, their orientations, and how different spaces within the home connect."
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are an expert in describing spatial relations within indoor environments, especially homes."},
                #{"role": "user", "content": prompt_message},
                {"role": "user", "content": prompt_message + " Text: " + prompt},
            ],
            max_tokens=400
        )

        summarized_text = response['choices'][0]['message']['content'].strip()
        return summarized_text
    except Exception as e:
        print(f'Error occurred while summarizing. Error: {e}')
        return "Error in summarization."

def generate_score(folder_path, file_name1, file_name2):
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                combined_text += file.read() + '\n'

    description_file_path1 = os.path.join(folder_path, file_name1.replace('.png', '.txt'))
    description_file_path2 = os.path.join(folder_path, file_name2.replace('.png', '.txt'))

    with open(description_file_path1, 'r') as file:
        description1 = file.read()

    with open(description_file_path2, 'r') as file:
        description2 = file.read()

    #prompt_part_a = "Rate the spatial relations between two images on a scale of 1 to 10. 0 point pair example: 'A room with a bed near a window' and 'A kitchen with a dining table near a window'. 10 point pair example: 'A kitchen with a table, sink, fridge, and microwave' and 'A well-lit kitchen with a sink, dishwasher, and granite counter'. "


    prompt_part_a = "Rate the camera view connectivity between two images on a scale of 1 to 10."
    prompt_part_b = f"Image 1: {description1}"
    prompt_part_c = f"Image 2: {description2}"
    prompt =  prompt_part_a + prompt_part_b + prompt_part_c+"Only Output: __(int)"
    prompt_length = len(prompt)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a spatial reasoning evaluator."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2
        )

        result_text = response['choices'][0]['message']['content'].strip()
        score = re.search(r'\d+', result_text)
        if score is not None:
            score_value=int(score.group())
            if 0 <= score_value <= 10:
                return score_value, prompt_length
            else:
                return 5, prompt_length
        else:
            return 5, prompt_length

    except Exception as e:
        print(f'TimeoutError occurred for {file_name1} and {file_name2}. Skipping and continuing.')
        return 5, prompt_length  # return default score and prompt length

def generate_scores_for_subdir(subdir_path):
    global chars_sent_this_minute
    global current_minute

    file_names = os.listdir(subdir_path)
    n_files = len(file_names)
    scores = np.empty((0, 5), dtype=object)  # 更新列数为5

    for i in range(n_files):
        file_name1 = file_names[i]
        if not file_name1.endswith('.png'):
            continue
        for j in range(i + 1, n_files):
            file_name2 = file_names[j]
            if not file_name2.endswith('.png'):
                continue

            while chars_sent_this_minute >= 10000 and (time.time() // 60) == current_minute:
                time.sleep(1)
            if (time.time() // 60) != current_minute:
                chars_sent_this_minute = 0
                current_minute = time.time() // 60

            score, prompt_length = generate_score(subdir_path, file_name1, file_name2)
            chars_sent_this_minute += prompt_length

            print("current folder"+subdir_path)
            print("score" + str(score))
            print("prompt_length" + str(prompt_length))
            print("first pic" + file_name1 + " sec pic" + file_name2)

            # 提取图像文件名中的数字
            first_image_number = re.search(r'\d+', file_name1).group()
            second_image_number = re.search(r'\d+', file_name2).group()

            scores = np.vstack((scores, [folder, subdir, first_image_number, second_image_number, score]))  # 更新格式

    return scores

for folder in folderList:
    folder_path = './dataset1/' + folder
    subdirs = [str(d) for d in range(6)]
    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir, 'saved_obs')

        # 检查子文件夹是否存在
        if os.path.exists(subdir_path):
            scores = generate_scores_for_subdir(subdir_path)
            np.save(os.path.join(subdir_path, 'scores-alter.npy'), scores)
            pd.DataFrame(scores, columns=['Folder', 'Subdir', 'first image', 'second image', 'Score']).to_csv(os.path.join(subdir_path, 'scores-alter.csv'), index=False)
        else:
            print(f'Subfolder {subdir} does not exist in {folder}. Skipping.')
