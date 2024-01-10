from supabase import Client, create_client
import pandas as pd
import json
import numpy as np
import openai 
# from diffusers import DiffusionPipeline
# import torch
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import OpenAIModerationChain, SequentialChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
import cv2
from PIL import Image
from PIL import ImageDraw, ImageFont
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from datetime import datetime
import logging
from dotenv import load_dotenv, find_dotenv

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

load_dotenv()


logger = logging.getLogger(__name__)
log_file_path = f'/app/app/logs/asset_{timestamp}.log'
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_bucket = 'solarplexus'
SUPABASE_HOST= os.environ.get("SUPABASE_HOST")
SUPABASE_PASSWORD= os.environ.get("SUPABASE_PASSWORD")



try:

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info('Supabase connection successfull')

    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16")
    # commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")


except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    logger.error(e)


def fetch_table_data(table_name):
    table_response = supabase.from_(table_name).select("*").execute()
    table_data_raw = [record for record in table_response.data]

    print(table_data_raw)
    # table = "segment_metadata"
    #         updates = {
    #             "process_id" : process_id
    #         }
    #         response_update = supabase.from_(table).update(updates).eq("id", segment_id).execute()

    #         update_d = [record for record in response_update.data]
    table_data_json = json.dumps(table_data_raw)
    table= json.loads(table_data_json)
    # update_segment_id = update[0]['id']
    # print(update_segment_id)

    return table


def generate_text(cluster_description, categorical_description, asset_to_individulize_file, tone_of_voice):
    prompt = f"""generate a different marketing asset image for a brand Solarplexus, to target big industries, 
    text should have font as Raleway and secondary font as Open Sans, image should have primary 
    colour code should be #EB691B and secondary color code should be #4B4B4B, and generate images based on these cluster
    description {cluster_description} {categorical_description}"""

    
    
    # response = openai.ChatCompletion.create(
    #     # model="gpt-3.5-turbo-0613",
    #     model= "gpt-4",
    #     # model = "gpt-3.5-turbo-16k",
    #     # model="gpt-4-0613",
    #     messages=[

    #             {"role": "user",
    #                 "content": f"""generate a very professional marketing asset 
    #                 text for a brand to place on the image, to target big financial industries the text generate should be
    #                 very specific and should be based on these descriptions {cluster_description} {categorical_description} that it will be targetted for."""},

    #         ]
    #     )

    # result = ''
    # for choice in response.choices:
    #     result += choice.message.content

    # print(result)














    reader = PdfReader(asset_to_individulize_file)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    print(raw_text[:100])

    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)



    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    llm_answer_list = []

    ques = f"""convert the result to short summary in one line based on the cluster description {cluster_description} and {categorical_description}, 
            make the results proper and different based on the industry the cluster is focused and the answer should be very clear. The tone of the document
            should be in {tone_of_voice}. Don't give me anything else. The result should be attractive that can be used for marketing campaigns."""

    docs = docsearch.similarity_search(ques)
    llm_answer = chain.run(input_documents=docs, question=ques)

    print("llm_answer----------->", llm_answer)








    return llm_answer


def get_extracted_data(extraction_id_brand, extraction_id_tone):

    query = supabase.from_("data_extraction").select("*").eq("extraction_id", extraction_id_brand).execute()
    update_d = [record for record in query.data]

    print(update_d)
    color_data = json.dumps(update_d)
    color = json.loads(color_data)
    llm_answer = color[0]["llm_answer"]

    print(type(llm_answer))
    print(llm_answer)
    answer =  json.loads(llm_answer)

    # Create variables to store answers
    primary_color = None
    brand_name = None
    primary_font = None
    secondary_color = None
    secondary_font = None

    # Process the list of dictionaries
    for item in answer:
        question = item['question']
        answer = item['answer']

        if "primary colour code" in question:
            primary_color = answer
        elif "brand name" in question:
            brand_name = answer
        elif "primary font" in question:
            primary_font = answer
        elif "secondary colour code" in question:
            secondary_color = answer
        elif "secondary font" in question:
            secondary_font = answer

    # Print the stored answers
    print("Primary Color:", primary_color)
    print("Brand Name:", brand_name)
    print("Primary Font:", primary_font)
    print("Secondary Color:", secondary_color)
    print("Secondary Font:", secondary_font)


    response = supabase.from_("data_extraction").select("*").eq("extraction_id", extraction_id_tone).execute()
    response_d = [record for record in response.data]

    print(response_d)
    tone_data = json.dumps(response_d)
    tone = json.loads(tone_data)
    tone_llm_answer = tone[0]["llm_answer"]

    print(type(tone_llm_answer))
    print(tone_llm_answer)
    tone_answer =  json.loads(tone_llm_answer)

    # Create variables to store answers
    tone_of_voice = None

    # Process the list of dictionaries
    for item in tone_answer:
        question = item['question']
        answer = item['answer']

        if "tone of voice" in question:
            tone_of_voice = answer

    # Print the stored answers
    print("tone of voice:", tone_of_voice)





    return {"primary_color": primary_color, "secondary_color": secondary_color, "primary_font": primary_font, "secondary_font":secondary_font, "brand_name": brand_name, "tone_of_voice": tone_of_voice}
    

def get_rgb_colors(primary_color, secondary_color):
    rgb_color = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo-0613",
            model= "gpt-4",
            # model = "gpt-3.5-turbo-16k",
            # model="gpt-4-0613",
            messages=[

                    {"role": "user",
                        "content": f"""Generate RGB of color {primary_color} and color {secondary_color} and give me a json format strictly only in Red Green Blue nested dictionary and nothing else.
                                    You can consider this as an example to generate you result: 
                                    EXAMPLE: """ + """{"EB691B": { "Red": 235,"Green": 105"Blue": 27},"4B4B4B": { "Red": 75,"Green": 75,"Blue": 75},"95CDED": {"Red": 149,"Green": 205, "Blue": 237}}"""},

                ]
            )

    rgb_result = ''
    for choice in rgb_color.choices:
        rgb_result += choice.message.content

    print(rgb_result)
    print(type(rgb_result))
    
    
    
    "------------------------covert to json------------------------------"
    
    colors = json.loads(rgb_result)
    print(colors)
    print(type(colors))
    
    "------------------------reading rgb from json------------------------"
    

    # Initialize variables for primary and secondary colors
    primary_color_rgb = ()
    secondary_color_rgb = ()

    # Iterate through the dictionary and store RGB values for the first two keys
    for idx, (key, rgb_values) in enumerate(colors.items()):
        if idx == 0:
            primary_color_rgb = (rgb_values['Red'], rgb_values['Green'], rgb_values['Blue'])
        elif idx == 1:
            secondary_color_rgb = (rgb_values['Red'], rgb_values['Green'], rgb_values['Blue'])
        else:
            break  # Only store values for the first two keys

    # Print the stored RGB values
    print(f"Primary Color: {primary_color_rgb}")
    print(f"Secondary Color: {secondary_color_rgb}")  

    return {"primary_color_rgb": primary_color_rgb, "secondary_color_rgb": secondary_color_rgb}  


def fetch_background_image(file_id_background_image):
    type = "picture_bank"
    user = supabase.from_("file_data").select("*").eq("id",file_id_background_image).eq("type", type).execute()

    user_data = [record for record in user.data]
    print("user_data",user_data)
    data = json.dumps(user_data)
    d = json.loads(data)

    file_path = d[0]["path"]
    file_type = d[0]["type"]



    try:


        local_file_path = f'/app/app/services/files/{file_path.split("/")[-1]}'
        print(local_file_path)

        print(file_path)
        with open(local_file_path, 'wb+') as f:
            data = supabase.storage.from_(supabase_bucket).download(file_path)
            f.write(data)

    except Exception as e:

        logging.error('An error occurred:', exc_info=True)

    return local_file_path

# fetch_background_image(803)


    

def fetch_logo(file_id_log):
    type = "logo"
    user = supabase.from_("file_data").select("*").eq("id",file_id_log).eq("type", type).execute()

    user_data = [record for record in user.data]
    print("user_data",user_data)
    data = json.dumps(user_data)
    d = json.loads(data)

    file_path = d[0]["path"]
    file_type = d[0]["type"]


    try:


        local_file_path = f'/app/app/services/files/{file_path.split("/")[-1]}'
        print(local_file_path)

        print(file_path)
        with open(local_file_path, 'wb+') as f:
            data = supabase.storage.from_(supabase_bucket).download(file_path)
            f.write(data)

    except Exception as e:

        logging.error('An error occurred:', exc_info=True)

    return local_file_path


def fetch_asset_individualize(project_id):
    group = "asset"
    user = supabase.from_("project_files").select("*").eq("project_id",project_id).eq("group", group).execute()

    user_data = [record for record in user.data]
    print("user_data",user_data)
    data = json.dumps(user_data)
    d = json.loads(data)

    file_path = d[0]["path"]
    file_group = d[0]["group"]


    try:


        local_file_path = f'/app/app/services/files/{file_path.split("/")[-1]}'
        print(local_file_path)

        print(file_path)
        with open(local_file_path, 'wb+') as f:
            data = supabase.storage.from_(supabase_bucket).download(file_path)
            f.write(data)

    except Exception as e:

        logging.error('An error occurred:', exc_info=True)

    return local_file_path

def combine_text_image(cluster_id, background_image_path, logo_path, asset_to_individualize, primary_color_rgb, secondary_color_rgb):
    base_image = Image.open(background_image_path)

    # Initialize the drawing context
    draw = ImageDraw.Draw(base_image)

    # Set primary and secondary colors
    primary_color_rgb = primary_color_rgb  # (R, G, B) for #EB691B
    secondary_color_rgb = secondary_color_rgb  # (R, G, B) for #4B4B4B

    # Yellow C100%, Pantone 281 C100%

    # Use built-in fonts
    primary_font = ImageFont.load_default()  # Use the default font
    secondary_font = ImageFont.load_default()  # Use the default font

    # Set the text to be displayed
#     text = "Empower Your Legacy Giants with our premier solutions. Captivating 8,200+ financial industries and counting, our robust marketing tools are uniquely designed to serve your distinct needs. Embrace efficiency, cultivate growth and be a part of the top-financial trendsetters across the United Kingdom. Propel your business forward in a landscape dominated by Kingsley Napley LLP and others. Join the movement - Experience difference with us."
    
    # text = result
    text = asset_to_individualize
    
    # Set the text position for the primary color
    text_position_primary = (20, 80)

    # Draw text in primary color with default font
    draw.text(text_position_primary, text, fill=primary_color_rgb, font=primary_font)

    # Load the overlay image
    # logo = Image.open("arkitektkopia-loggo-ritsPP-cmyk.png")
    logo = Image.open(logo_path)

    # You may need to resize the overlay image to fit
    logo = logo.resize((80, 50))  # Adjust the size as needed

    # Paste the overlay image on top of the base image
    base_image.paste(logo, (400, 20))

    # Save the modified image
    asset_path = f"asset_{cluster_id}.jpg"
    base_image.save(asset_path)

    # Display the modified image
    # base_image.show()

    return asset_path





# def combine_text_image(cluster_id, background_image_path, logo_path, asset_to_individualize, primary_color_rgb, secondary_color_rgb):
#     base_image = Image.open(background_image_path)

#     draw = ImageDraw.Draw(base_image)

#     primary_color_rgb = primary_color_rgb
#     font_size = 20

#     # Use the truetype font
#     primary_font = ImageFont.load_default()  # Use the default font
#     secondary_font = ImageFont.load_default()

#     font = ImageFont.truetype(primary_font, font_size)

#     text = asset_to_individualize
#     text_width, text_height = primary_font.getsize(text)
#     text_x = (base_image.width - text_width) // 2
#     text_y = (base_image.height - text_height) // 2

#     draw.text((text_x, text_y), text, fill=primary_color_rgb, font=primary_font)


#     logo = Image.open(logo_path)
#     logo = logo.resize((80, 50))

#     base_image.paste(logo, (400, 20))

#     asset_path = f"asset_{cluster_id}.jpg"
#     base_image.save(asset_path)

#     return asset_path





 
def asset_creation(table_name, user_id, project_id, extraction_id_brand, extraction_id_tone, file_id_log, file_id_background_image):

    print("entered")
    process_id = None
    try:
        process_data_insert = [
                    {
                        "user_id" :user_id,
                        "process_type": "asset_creation",
                        "process_status": "in_progress",
                        "start_at" : datetime.now().isoformat()
                    },
                ]


        process= supabase.from_("process").insert(process_data_insert).execute()
        process_data = [record for record in process.data]

        p_data = json.dumps(process_data)
        p = json.loads(p_data)
        process_id = p[0]["process_id"]

        print("process table:*******************", p)




        # table_name = "segment_47b0ffec-356a-4c35-8704-23b153d345c5_1087"
        
        "--------------------------------------------------------------------"

        """# Read data from Supabase
        query = f"SELECT * FROM {table_name}"
        response = supabase.from_(table_name).select("*").execute()
        update_d = [record for record in response.data]

        print(update_d)
        # table = "segment_metadata"
        #         updates = {
        #             "process_id" : process_id
        #         }
        #         response_update = supabase.from_(table).update(updates).eq("id", segment_id).execute()

        #         update_d = [record for record in response_update.data]
        response_u = json.dumps(update_d)
        update= json.loads(response_u)
        update_segment_id = update[0]['id']
        print(update_segment_id)"""

        "--------------------------------------------------------------------"

        background_image_path = fetch_background_image(file_id_background_image)
        logo_path = fetch_logo(file_id_log)

        table = fetch_table_data(table_name)
        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(table)




        # Group the data by the cluster column
        cluster_column = "Cluster"

        grouped_clusters = df.groupby(cluster_column)

        categorical_columns = df.select_dtypes(exclude=[np.number])

        result_filenames = []
        asset_id = []
        asset_path = []

        for cluster_id, cluster_data in grouped_clusters:
            # Perform operations on cluster_data
            # You can access each cluster's data using cluster_data
            # For example, to get the description of the cluster:
        #     cluster_description = cluster_data["description"].iloc[0]
        #     print(f"Cluster Name: {cluster_name}, Description: {cluster_description}")

                    
            # Descriptive statistics for each cluster
            cluster_description = df[df['Cluster'] == cluster_id].describe()
            print(cluster_description)


            # Descriptive statistics for categorical columns
            categorical_cluster_data = categorical_columns[df['Cluster'] == cluster_id]
            categorical_description = categorical_cluster_data.describe()
        #     print("Categorical Column Statistics:")
        #     print(categorical_description)

            print(f"Cluster Name: {cluster_id} {cluster_description} {categorical_description}")
            
            
            
            "--------------------------------------------------------------------"
            
            # prompt = f"""generate a marketing asset image for a brand Solarplexus, to target big industries, 
            # text should have font as Raleway and secondary font as Open Sans, image should have primary 
            # colour code should be #EB691B and secondary color code should be #4B4B4B, and generate images based on these cluster
            # description {cluster_description} {categorical_description}"""
            # image = pipe(prompt).images[0]
            # print(image)
            # filename = f'result_{cluster_id}.jpg'
            # image.save(filename)
            # print(filename)
            # result_filenames.append(filename)
            
            
            
            # """response = openai.ChatCompletion.create(
            #     # model="gpt-3.5-turbo-0613",
            #     model= "gpt-4",
            #     # model = "gpt-3.5-turbo-16k",
            #     # model="gpt-4-0613",
            #     messages=[

            #             {"role": "user",
            #                 "content": f"""generate a very professional marketing asset 
            #                 text for a brand to place on the image, to target big financial industries the text generate should be
            #                 very specific and should be based on these descriptions {cluster_description} {categorical_description} that it will be targetted for."""},

            #         ]
            #     )

            # result = ''
            # for choice in response.choices:
            #     result += choice.message.content

            # print(result)"""
            
            "--------------------------------------------------------------------"
            
            extracted_data = get_extracted_data(extraction_id_brand, extraction_id_tone)



            primary_color = extracted_data["primary_color"]
            secondary_color = extracted_data["secondary_color"]
            primary_font = extracted_data["primary_font"]
            secondary_font = extracted_data["secondary_font"]
            brand_name = extracted_data["brand_name"]
            tone_of_voice = extracted_data["tone_of_voice"]



            asset_to_individulize_file = fetch_asset_individualize(project_id)

            
            asset_to_individualize = generate_text(cluster_description, categorical_description, asset_to_individulize_file, tone_of_voice)
            




            "--------------------------------------------------------------------"
            
            "------------------------get color from db----------------------------"


            
            # extraction_id = 789
            
            """query = supabase.from_("data_extraction").select("*").eq("extraction_id", extraction_id).execute()
            update_d = [record for record in query.data]

            print(update_d)
            color_data = json.dumps(update_d)
            color = json.loads(color_data)
            llm_answer = color[0]["llm_answer"]

            print(type(llm_answer))
            print(llm_answer)
            answer =  json.loads(llm_answer)

            # Create variables to store answers
            primary_color = None
            brand_name = None
            primary_font = None
            secondary_color = None
            secondary_font = None

            # Process the list of dictionaries
            for item in answer:
                question = item['question']
                answer = item['answer']

                if "primary colour code" in question:
                    primary_color = answer
                elif "brand name" in question:
                    brand_name = answer
                elif "primary font" in question:
                    primary_font = answer
                elif "secondary colour code" in question:
                    secondary_color = answer
                elif "secondary font" in question:
                    secondary_font = answer

            # Print the stored answers
            print("Primary Color:", primary_color)
            print("Brand Name:", brand_name)
            print("Primary Font:", primary_font)
            print("Secondary Color:", secondary_color)
            print("Secondary Font:", secondary_font)"""
            
            "--------------------------------------------------------------------"





            
            
            "--------------------------generate rgb color-------------------------"
            
            
            # primary = "Yellow C100%, Pantone 281 C100%"
            # secondary = ""





            "--------------------------------------------------------------------"


            # rgb_color = openai.ChatCompletion.create(
            #         # model="gpt-3.5-turbo-0613",
            #         model= "gpt-4",
            #         # model = "gpt-3.5-turbo-16k",
            #         # model="gpt-4-0613",
            #         messages=[

            #                 {"role": "user",
            #                     "content": f"""Generate RGB of color {primary_color} and color {secondary_color} and give me a json format in Red Green Blue nested dictionary and nothing else"""},

            #             ]
            #         )

            # rgb_result = ''
            # for choice in rgb_color.choices:
            #     rgb_result += choice.message.content

            # print(rgb_result)
            # print(type(rgb_result))
            
            
            
            # "------------------------covert to json------------------------------"
            
            # colors = json.loads(rgb_result)
            # print(colors)
            # print(type(colors))
            
            # "------------------------reading rgb from json------------------------"
            

            # # Initialize variables for primary and secondary colors
            # primary_color_rgb = ()
            # secondary_color_rgb = ()

            # # Iterate through the dictionary and store RGB values for the first two keys
            # for idx, (key, rgb_values) in enumerate(colors.items()):
            #     if idx == 0:
            #         primary_color_rgb = (rgb_values['Red'], rgb_values['Green'], rgb_values['Blue'])
            #     elif idx == 1:
            #         secondary_color_rgb = (rgb_values['Red'], rgb_values['Green'], rgb_values['Blue'])
            #     else:
            #         break  # Only store values for the first two keys

            # # Print the stored RGB values
            # print(f"Primary Color: {primary_color_rgb}")
            # print(f"Secondary Color: {secondary_color_rgb}")
            
            

            "--------------------------------------------------------------------"


            rgb_colors = get_rgb_colors(primary_color, secondary_color)

            primary_color_rgb = rgb_colors['primary_color_rgb']
            secondary_color_rgb = rgb_colors['secondary_color_rgb']
            
            
            

            "--------------------------------------------------------------------"
            
            
            "------------------------reading image----------------------"

            """# filename = f'result_{cluster_id}.jpg'
            
            # Load the existing image
            base_image = Image.open(background_image_path)

            # Initialize the drawing context
            draw = ImageDraw.Draw(base_image)

            # Set primary and secondary colors
            primary_color_rgb = primary_color_rgb  # (R, G, B) for #EB691B
            secondary_color_rgb = secondary_color_rgb  # (R, G, B) for #4B4B4B

            # Yellow C100%, Pantone 281 C100%

            # Use built-in fonts
            primary_font = ImageFont.load_default()  # Use the default font
            secondary_font = ImageFont.load_default()  # Use the default font

            # Set the text to be displayed
        #     text = "Empower Your Legacy Giants with our premier solutions. Captivating 8,200+ financial industries and counting, our robust marketing tools are uniquely designed to serve your distinct needs. Embrace efficiency, cultivate growth and be a part of the top-financial trendsetters across the United Kingdom. Propel your business forward in a landscape dominated by Kingsley Napley LLP and others. Join the movement - Experience difference with us."
            
            # text = result
            text = asset_to_individualize
            
            # Set the text position for the primary color
            text_position_primary = (100, 100)

            # Draw text in primary color with default font
            draw.text(text_position_primary, text, fill=primary_color_rgb, font=primary_font)

            # Load the overlay image
            # logo = Image.open("arkitektkopia-loggo-ritsPP-cmyk.png")
            logo = Image.open(logo_path)

            # You may need to resize the overlay image to fit
            logo = logo.resize((100, 100))  # Adjust the size as needed

            # Paste the overlay image on top of the base image
            base_image.paste(logo, (200, 200))

            # Save the modified image
            base_image.save(f"asset_{cluster_id}.jpg")

            # Display the modified image
            base_image.show()"""


            "--------------------------------------------------------------------"


            local_asset_path = combine_text_image(cluster_id, background_image_path, logo_path, asset_to_individualize, primary_color_rgb, secondary_color_rgb)

            bucket_path = f"/asset/{user_id}/{project_id}/asset_{cluster_id}.jpg"

                    # print("Bucket Pathhhhhhhhhhhhhhh", bucket_path)
            with open(local_asset_path, 'rb') as f:
                supabase.storage.from_(supabase_bucket).upload(file=f,path=bucket_path)

            


    
                
            asset_data_insert = [
                        {
                            "user_id" :user_id,
                            "project_id": project_id,
                            "asset_path": bucket_path
                        },
                    ]


            asset= supabase.from_("asset_metadata").insert(asset_data_insert).execute()
            asset_data = [record for record in asset.data]

            p_data = json.dumps(asset_data)
            p = json.loads(p_data)
            print("asssettttttt", p)
            assetid = p[0]["id"]
            print("Asset id---------", assetid)

            asset_id.append(assetid)
            asset_path.append(bucket_path)

            print("process table:*******************", p)
    
    



            process_data_update = {
                            "process_status": "stopped",
                            "end_at" : datetime.now().isoformat()
                        }
            supabase.from_("process").update(process_data_update).eq("process_id", process_id).execute()

            logger.info(f"asset creation done for segment {cluster_id}")

        os.remove(local_asset_path)
        os.remove(background_image_path)
        os.remove(logo_path)
        logger.info("asset creation done")



        return {"asset_id": asset_id, "asset_path": asset_path}
    
    except Exception as e:
        logger.error(e)
        print(e)
        return {"error": e, "status":"error"}