import os
import json
import glob
import cv2
import numpy as np
import requests
import base64
from pdf2image import convert_from_path
from ultralyticsplus import YOLO, render_result
from pdf2image import convert_from_path
import ast
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index import SimpleDirectoryReader
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
import os
from pydantic import BaseModel
import pandas as pd
from joblib import Parallel, delayed

# Open AI Key
open_ai_key =  "******************************************************"

###################################################################################################
# Prompts
###################################################################################################

# Defining the structure of the Metadata
class Tables(BaseModel):
    """Data model for a Tables."""
    # title: str
    # category: str
    # discount: str
    # price: str
    # rating: str
    # review: str
    # description: str
    # inventory: str
    title : str
    header : list
    # subheader : list
    rows : list
    description: str    
    
OPENAI_API_TOKEN = "******************************************************"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=10000
)


###################################################################################################
# Prompts
###################################################################################################


# prompt_template_str = """
#     If the given image contain tables then understand the table and return the answer in json format else just provide None.
#         For example, if the image shows a table with the following headers and subheaders:
#         | System | Development | Test |
#         | EM% | EX% | EM% | EX% |

#         The output should be:

#         System | Development EM% | Development EX% | Test EM% | Test EX%

#         The rows should be in a list format, such as ['BRIDGE v2 + BERT', 71.1, 70.3, 67.5, 68.3].
# """

prompt_template_str = """
    Summarize the image, if it contains a table, otherwise provide None.
    
    below are few examples,
    
    Example 1:
    If the image shows a table with the following headers and subheaders:
    | System | Development | Test |
    | EM% | EX% | EM% | EX% |

    The output should be:

    System | Development EM% | Development EX% | Test EM% | Test EX%

    The rows should be in a list format and in the same order, such as ['BRIDGE v2 + BERT', 71.1, 70.3, 67.5, 68.3].

    Please don't make assumptions and return the answer in JSON format.
    """


#############################################################################################
# ImageStore
#############################################################################################

def createImageStore(source_path):
    folder_path = source_path  # Specify the folder directory containing the PDF files
    image_store_folder = r"./DocData/ImageStore"  # Specify the parent image store folder path
    
    # Create the parent image store folder if it doesn't exist
    os.makedirs(image_store_folder, exist_ok=True)
    
    # Iterate through the PDF files in the folder directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            print(f"Processing ---> {file_name}")
            
            # Create the subfolder with the file name inside the image store folder
            subfolder_path = os.path.join(image_store_folder, os.path.splitext(file_name)[0])
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            else:
                # If subfolder already exists, move to the next file
                print("Images are already Existing")
                continue
            
            
            # Convert PDF to images and save them inside the subfolder
            source_path = os.path.join(folder_path, file_name)
            images = convert_from_path(source_path, poppler_path=r'C:\Program Files\poppler-23.11.0\Library\bin')
            for i, image in enumerate(images):
                image_path = os.path.join(subfolder_path, f"{os.path.splitext(file_name)[0]}_page{i}.jpg")
                image.save(image_path, 'JPEG')
            
        

    
    return True


################################################################################################
# TableStore
################################################################################################

def createTableStore(image_store):
    # Load model
    model = YOLO('keremberke/yolov8m-table-extraction')
    # Set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.47  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    
    # Directory paths
    image_store_folder = image_store
    table_store_folder = r"./DocData/TableStore"
    if not os.path.exists(table_store_folder):
        os.makedirs(table_store_folder)
        
    # Iterate through the subfolders in the ImageStore
    for subfolder in os.listdir(image_store_folder):
        subfolder_path = os.path.join(image_store_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print("Extracting Tables")
            # Check if the corresponding folder exists in TableStore
            table_subfolder_path = os.path.join(table_store_folder, subfolder)
            if os.path.exists(table_subfolder_path):
                print("Table Folder already Existing")
                # If the folder exists in TableStore, skip processing and move to the next folder
                continue
            
            # Create corresponding subfolder in TableStore
            os.makedirs(table_subfolder_path)
            
            # Process each image in the subfolder
            pattern = "*.jpg"
            files = glob.glob(os.path.join(subfolder_path, pattern))
            for file in files:
                if os.path.isfile(file):  # Check if the file is a regular file
                    image = file
                    # Perform inference
                    results = model.predict(image)
                    # Observe results
                    print(results[0].boxes)
                    render = render_result(model=model, image=image, result=results[0])
                    boxes = results[0].boxes.xyxy.tolist()
                    table_count = 1
                    for i in range(len(boxes)):
                        first_box = boxes[i]
                        table_cord = first_box[:4]
                        print(table_cord)
                        cropped_img = np.asarray(render.crop(table_cord))
                        if cropped_img.dtype != np.uint8:
                            cropped_img = cropped_img.astype(np.uint8)
                        _, encoded_img = cv2.imencode('.jpg', cropped_img)
                        
                        # Get the file name and page number
                        file_name = os.path.splitext(os.path.basename(file))[0]
                        page_number = i
                        # Add "table" keyword to the file name
                        table_name = f"{file_name}_table{table_count}.jpg"
                        # Save the cropped image with the desired format
                        table_image_path = os.path.join(table_subfolder_path, table_name)
                        with open(table_image_path, 'wb') as f:
                            f.write(encoded_img)
                        table_count += 1
    return True


###################################################################################################
# Summarizing the Tables
###################################################################################################

def process_image(file, subfolder_path): #, json_data):
    # Check if text file already exists
    image_name = os.path.splitext(os.path.basename(file))[0]
    
    metadata_file = os.path.join(subfolder_path, f"{image_name}")
    
    # if metadata_file in json_data.keys():
    #     return metadata_file, None

    # Rest of your code to extract tables and create summary_dict
    try:
        # Code to extract tables from the image file
        table_image_documents = SimpleDirectoryReader(input_files=[file]).load_data()
        openai_program_amazon = MultiModalLLMCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(Tables),
            image_documents=[table_image_documents[0]],
            prompt_template_str=prompt_template_str,
            llm=openai_mm_llm,
            verbose=True,
        )
        
        response = openai_program_amazon()
        # for res in response:
        #     print(res)

        # Extract page number and page name
        page_number = image_name.split("_")[1][4:]  # Remove the "page" prefix
        page_name = image_name.split("_")[0]

        if response is None:
            summary_dict = {
                "header": None,
                "rows": None,
                "page_no": page_number,
                "file_name": page_name,
                "image_path": file
            }
            
        else:
            # Create dictionary with summary, page number, page name, and image path
            summary_dict = {
                "header": response.header,
                "rows": response.rows,
                "description" : response.description,
                "page_no": page_number,
                "file_name": page_name,
                "image_path": file
            }

        return metadata_file, summary_dict

    except Exception as e:
        print(e)
        return metadata_file, None



###################################################################################################
# Parallel Computation for Summarizing the Tables
###################################################################################################

def metadataCreation(table_store_folder):
    
    # Specify the path to your JSON file
    
    json_summary_dir_path = "./DocData/SummaryJson"
    os.makedirs(json_summary_dir_path, exist_ok=True)
    json_files = os.listdir('./DocData/SummaryJson')
    
    #-------------------------------------------------------
    # json_file_path = r"./DocData/SummaryJson/metadata.json"
    # if not os.path.exists(json_file_path):
    #     temp_dict = {}
    #     with open(json_file_path, 'w') as fp:
    #         json.dump(temp_dict)
    # Read the JSON file
    # with open(json_file_path, "r") as f:
    #     json_data = json.load(f)
    #--------------------------------------------------------
    
        
    # Iterate through the subfolders in the TableStore
    for subfolder in os.listdir(table_store_folder):
        subfolder_path = os.path.join(table_store_folder, subfolder)
            
        # Process each image in the subfolder
        pattern = "*.jpg"
        files = glob.glob(os.path.join(subfolder_path, pattern))
        
        
        if subfolder+'.json' not in json_files:
            print(f"Creating metadata for {subfolder}")
            
            # Perform parallel computation for image processing
            summaries = Parallel(n_jobs=8, prefer="threads")(
                delayed(process_image)(file, subfolder_path) for file in files
            )
            
            # Update the summary_log with the results
            summary_log = {metadata_file: summary_dict for metadata_file, summary_dict in summaries}
            
            with open(os.path.join(json_summary_dir_path,subfolder+'.json'), 'w') as f:
                json.dump(summary_log, f)
            
        
        
    return json_summary_dir_path


###################################################################################################
# Main Function
###################################################################################################

def main(source_path = r"./DocData/SourceDirectory"):
    list_jsons = []
    try: 
        imageStoreStatus =  createImageStore(source_path)
        if imageStoreStatus == True:
            tableStoreStatus = createTableStore(image_store=r"./DocData/ImageStore")
            
            if tableStoreStatus == True:
                summary_folder_path = metadataCreation(table_store_folder=r"./DocData/TableStore")
            
        list_jsons = [i[:-4]+'.json' for i in os.listdir(source_path) if i.endswith('.pdf')]
    except Exception as e:
        print(e)
    
    return list_jsons


result = main()

print("List of Files Passed : ", result)


## Metadata way

# if len(summary_log.keys()) > 0:
    
#     # Specify the path to your JSON file
#     json_file_path = r"./DocData/SummaryJson/metadata.json"
    
#     # Read the JSON file
#     with open(json_file_path, "r") as f:
#         json_data = json.load(f)
        
#     for key, value in summary_log.items():
#         if key in json_data.keys() and value['header'] == None:
#             continue
#         else:
#             json_data[key] = value
            
#     # Store the summary_dict_list as metadata
#     with open('metadata.json', 'w') as f:
#         json.dump(json_data, f)


##### Meta data creation old

# def metadataCreation(table_store_folder):
#     # Specify the path to your JSON file
#     json_file_path = r"./DocData/metadata.json"

#     # Read the JSON file
#     with open(json_file_path, "r") as f:
#         json_data = json.load(f)

#     # Iterate through the subfolders in the TableStore
#     for subfolder in os.listdir(table_store_folder):
#         subfolder_path = os.path.join(table_store_folder, subfolder)
        
#         if os.path.isdir(subfolder_path):
#             # Process each image in the subfolder
#             pattern = "*.jpg"
#             files = glob.glob(os.path.join(subfolder_path, pattern))
#             summary_log = {}
            
#             for file in files:
#                 if os.path.isfile(file):
#                     # Check if text file already exists
#                     image_name = os.path.splitext(os.path.basename(file))[0]
#                     metadata_file = os.path.join(subfolder_path, f"{image_name}")
                    
#                     if metadata_file in json_data.keys():
#                         continue
                    
#                     # if os.path.isfile(metadata_file):
#                     #     continue 
#                     # extracting tables
#                     try:
#                         print("file", file)
#                         table_image_documents = SimpleDirectoryReader(input_files=[file]).load_data()
#                         openai_program_amazon = MultiModalLLMCompletionProgram.from_defaults(
#                             output_parser=PydanticOutputParser(Tables),
#                             image_documents = [table_image_documents[0]],
#                             prompt_template_str=prompt_template_str,
#                             llm=openai_mm_llm,
#                             verbose=True,
#                         )
#                         response = openai_program_amazon()
#                         for res in response:
#                             print(res)
                            
#                     except Exception as e:
#                         print(e)
#                         response = None
                        
#                     # Extract page number and page name
#                     page_number = image_name.split("_")[1][4:]  # Remove the "page" prefix
#                     page_name = image_name.split("_")[0]
                    
#                     try:
#                         if response == None:
#                             summary_dict = {
#                             "header": None,
#                             "rows": None,
#                             "page_no": page_number,
#                             "file_name": page_name,
#                             "image_path": file}
#                         else:                          
#                             # Create dictionary with summary, page number, page name, and image path
#                             summary_dict = {
#                                 "header": response.header,
#                                 "rows": response.rows,
#                                 "page_no": page_number,
#                                 "file_name": page_name,
#                                 "image_path": file
#                             }
#                     except Exception as e:
#                         print(e)
                        

#                     summary_log[metadata_file] = summary_dict
                    
#                     # # Store the summary as metadata
#                     # with open(f"{metadata_file}.json", "w") as f:
#                     #     json.dump(summary_dict, f)
                        
#     return summary_log

