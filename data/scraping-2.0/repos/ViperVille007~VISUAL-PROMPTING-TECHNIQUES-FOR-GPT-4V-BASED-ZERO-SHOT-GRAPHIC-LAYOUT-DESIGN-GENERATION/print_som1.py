import os
import pandas as pd
import os
from openai import OpenAI
import base64
import mimetypes
import re
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box
client = OpenAI(api_key="sk-xxxxxxxx")   ### enter your OpenAI API key here

experiment_name = 'GPT4V-GRID'  ###Replace with desired experiment name
path_ref = None
gt_boxes = [
[152, 704, 932, 828],
[183, 1046, 3297, 1629],
[823, 449, 1645, 634],
[132, 2051, 3306, 2945],
[124, 359, 1098, 644],
[132, 2051, 3306, 2945],
[142, 867, 936, 979],
[346, 440, 3455, 820],
[352, 911, 3078, 1257],
[172, 157, 1585, 394],
[352, 145, 1588, 333],
[148, 869, 1600, 1026],
[90, 321, 3365, 776],
[189, 741, 897, 834],
[186, 545, 1249, 882],
[355, 1598, 2862, 1960],
[108, 419, 957, 526],
[75, 765, 998, 923],
[419, 903, 2141, 1249],
[788, 389, 2668, 892],
[715, 1764, 2740, 2186],
[383, 138, 1353, 273],
[282, 819, 802, 908],
[583, 219, 1635, 447],
[119, 869, 961, 1001],
[293, 392, 1448, 577],
[698, 481, 2740, 1198],
[717, 750, 2721, 1121],
[911, 1351, 2535, 1571],
[514, 971, 2977, 1271],
[225, 1045, 3220, 1397],
[293, 689, 3100, 1156],
[1158, 2180, 2323, 2401],
[239, 840, 837, 944],
[805, 3028, 2650, 3249],
[682, 1686, 2783, 2057],
[295, 664, 769, 780],
[282, 430, 802, 550],
[1035, 208, 4130, 528],
[351, 1497, 3621, 2081],
[353, 435, 1405, 641],
[177, 503, 899, 654],
[101, 142, 799, 382],
[351, 1101, 3508, 1650],
[351, 898, 2904, 1186],
[364, 451, 1355, 767],
[310, 501, 1414, 715],
[814, 3090, 2632, 3346],
[364, 2384, 1829, 2772],
[920, 336, 2536, 618],
]

if experiment_name == 'GPT4V':
    path1 = 'inpainted_768'
elif experiment_name == 'GPT4V-SOM':
    path1 = 'som'
elif experiment_name == 'GPT4V-GRID':
    path1 = 'grid'
elif experiment_name == 'GPT4V-SLIC':
    path1 = 'slic'
elif experiment_name == 'GPT4V-WATER':
    path1 = 'watershed'
elif experiment_name == 'GPT4V-REFERENCE':
    path1 = 'inpainted_768'
    path_ref = 'reference'
csv_file_path = 'texts.csv'
og_image_folder = 'inpainted'

df = pd.read_csv(csv_file_path)
df['Text'] = df['Text'].astype(str)
text_column = df['Text']

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def run_experiment(experiment_name, path1, text_column, og_image_folder, path_ref):
    fscores = []
    
    for idx, image_filename in enumerate(os.listdir(og_image_folder)):

        image_path1 = os.path.join(path1, image_filename)
        # path_768 = os.path.join(path2, image_filename)

        text = text_column.iloc[idx]
        base64_image1 = encode_image(image_path1)
        og_img = Image.open(os.path.join(og_image_folder,image_filename))
        width, height = og_img.size
        
        if experiment_name != 'GPT4V-REFERENCE':
            prompt_gpt = ''' 
You are an expert design consultant with a creative mind.
You are provided with an image of dimensions 768*768 which is to be used for an advertisement banner. The objects/areas are labelled in the image.
Your task is to suggest the best place within the image to place the text of the advertisement '{text}'.
 
You must keep in mind the following things:
1. Position Selection: The given text is the most important text or the heade of an advertisement. Therefore, it has to positioned strictly in a plain area and must not overlap any other objects present in the image.
2. Font Size Selection: The text must have a font size which is large enough to garner attention, yet at the same time not overlap any of the objects in the image. So, estimate the font size for the header accordingly.
2. Font Color Selection: Identify and select the ideal font color as black or white according to the background. The font should be clearly visible and in contrast with the background.
 
You need to provide the ideal bounding box coordinates the text box for header. Calculate the area text would take and then give coordinates, as text should not go out of the image.
The width of the box would be x2-x1, so specify the coordinates according to the width it would need.
 
Your response must be in the following format:
Line1: 'Font Size = p'  where p is the font size suggestion.
Line2: 'Coordinates = [x1,y1] where x1, y1 are the top left coordinates of the bounding box.
Line3: 'Font Color = k' where k = 0,0,0 if font color is black and k = 256,256,256 if font color is white.
Line4: 'Bottom = [p1,q1]' where p1, q1 are the bottom right coordinates of the bounding box.
Just print these 4 lines and nothing else. There should be no space after any comma while printing coordinates.
 
Take a deep breath. Think step by step and answer.'''.format(text=text)
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_gpt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image1}",
                                    "detail": "high"
                                }
                            },
                            # {
                            #     "type": "image_url",
                            #     "image_url": {
                            #         "url": f"data:image/jpeg;base64,{base64_image2}",
                            #         "detail": "high"
                            #     }
                            # },
                        ],
                    }
                ],
                max_tokens=1000,
            )
        else:
            img_names = os.listdir(path_ref)
            img_name = random.choice(img_names)
            image_path_ref = os.path.join(path_ref, img_name)
            base64_image2 = encode_image(image_path_ref)
            prompt_gpt = ''' 
You are an expert design consultant with a creative mind.
You are provided with an image of dimensions 768*768 which is to be used for an advertisement banner. You are provided with a reference image which represents ideal textbox position.
Your task is to suggest the best place within the image to place the text of the advertisement '{text}'. 
 
You must keep in mind the following things:
1. Position Selection: The given text is the most important text or the heade of an advertisement. Therefore, it has to positioned strictly in a plain area and must not overlap any other objects present in the image.
2. Font Size Selection: The text must have a font size which is large enough to garner attention, yet at the same time not overlap any of the objects in the image. So, estimate the font size for the header accordingly.
2. Font Color Selection: Identify and select the ideal font color as black or white according to the background. The font should be clearly visible and in contrast with the background.
 
You need to provide the ideal bounding box coordinates the text box for header. Calculate the area text would take and then give coordinates, as text should not go out of the image.
The width of the box would be x2-x1, so specify the coordinates according to the width it would need.

Do remember that you have all the information required for the image analysis. 

Your response must be in the following format:
Line1: 'Font Size = p'  where p is the font size suggestion.
Line2: 'Coordinates = [x1,y1] where x1, y1 are the top left coordinates of the bounding box.
Line3: 'Font Color = k' where k = 0,0,0 if font color is black and k = 256,256,256 if font color is white.
Line4: 'Bottom = [p1,q1]' where p1, q1 are the bottom right coordinates of the bounding box.
Just print these 4 lines and nothing else. There should be no space after any comma while printing coordinates.
 
Take a deep breath. Think step by step and answer.
                         '''
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_gpt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image1}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image2}",
                                    "detail": "high"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )

        answer = response.choices[0].message.content
        print(image_filename)

        font_size_match = re.findall(r'Font Size = (\d+)', answer)
        coordinates_match = re.findall(r'Coordinates = \[(\d+),(\d+)\]', answer)
        font_size = int(font_size_match[0]) if font_size_match else None
        x1, y1 = map(int, coordinates_match[0]) if coordinates_match else (0, 0)
        # font_color_pattern = r'Font Color\s*=\s*([\w\s]+)'
        # font_color_matches = re.findall(font_color_pattern, text)
        # font_color = font_color_matches[0] if font_color_matches else None
        coordinates_match1 = re.findall(r'Bottom = \[(\d+),(\d+)\]', answer)
        p1, q1 = map(int, coordinates_match1[0]) if coordinates_match1 else (0, 0)
        rat = width/768
        rath = height/768
        fscores.append((int(x1*rat),int(y1*rath),int(p1*rat),int(q1*rath)))
    
        # rgb_values = re.findall(r'Font Color = (\d+),(\d+),(\d+)', answer)
        # if rgb_values:
        #     r, g, b = map(int, rgb_values[0])

        # input_image1 = Image.open(image_path1)

        # text_color=(r,g,b)   

        # font = ImageFont.truetype("extra/arial.ttf", size=font_size)
        # draw = ImageDraw.Draw(input_image1)

        # draw.text((x1, y1), text, font = font, fill=text_color, align ="left")
        # filename, _ = os.path.splitext(os.path.basename(image_filename))
        # output_filename = f"{filename}_processed768e.png"

        # output_folder_path = experiment_name +'/processed768e'
        # output_path = os.path.join(output_folder_path, output_filename)
        # input_image1.save(output_path)
    return np.array(fscores)

def calculate_iou(box1, box2):
    # Create Shapely boxes from bounding box coordinates
    shapely_box1 = box(box1[0], box1[1], box1[2], box1[3])
    shapely_box2 = box(box2[0], box2[1], box2[2], box2[3])

    # Calculate intersection and union
    intersection_area = shapely_box1.intersection(shapely_box2).area
    union_area = shapely_box1.union(shapely_box2).area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_giou(box1, box2):
    iou = calculate_iou(box1, box2)
    enclosing_box = box(min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3]))
    enclosing_area = enclosing_box.area
    giou = iou - (enclosing_area - iou) / enclosing_area
    return giou

def get_iou(gt_boxes, pred_boxes):
    iou_values = []
    giou_values = []
    for i in range(len(gt_boxes)):
        ground_truth_box = gt_boxes[i]
        result_box = pred_boxes[i]

        iou_values.append(calculate_iou(ground_truth_box, result_box))
        giou_values.append(calculate_giou(ground_truth_box, result_box))
    return iou_values #giou_values

all_boxes = []
all_ious = []
avgs= []
for i in range(3):
    pred_boxes = run_experiment(experiment_name, path1, text_column, og_image_folder, path_ref)
    all_boxes.append(pred_boxes)
    
    iou = get_iou(gt_boxes,pred_boxes)
    all_ious.append(iou)
    avgs.append(np.mean(iou))

max_iou = []
for i in range(len(all_ious[0])):
    max_iou.append(np.max([all_ious[0][i], all_ious[1][i], all_ious[2][i]]))

print(' For experiment {}: Average IoU: {}, Average of Average IoUs: {}, Best Iou: {}'.format(experiment_name, np.max(avgs), np.mean(avgs), np.mean(max_iou)))   

    



