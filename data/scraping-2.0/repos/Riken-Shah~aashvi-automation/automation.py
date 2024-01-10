import os
import uuid
from datetime import datetime
import pyperclip
import shutil
import gspread
import undetected_chromedriver as uc
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import requests
import json
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
import openai
import base64
import random
from utils import setup_openai, creds, get_gspread_client, generate_uuid, send_messages

setup_openai()
AUTOMATIC1111_URL = ""

client = gspread.authorize(creds)

gsheet = get_gspread_client()

driver = None


def upload_image_to_drive(filename, base64_image, folder_id):
    # decode the base64 image string
    image_data = base64.b64decode(base64_image)
    # write the image data to a file
    with open(f'/Users/rikenshah/Desktop/Fun/insta-model/{filename}', 'wb') as f:
        f.write(image_data)

    try:

        drive_service = build('drive', 'v3', credentials=creds)

        # create the file metadata
        file_metadata = {'name': filename, 'parents': [folder_id]}
        m = MediaFileUpload(f'/Users/rikenshah/Desktop/Fun/insta-model/{filename}', mimetype='image/png')

        # upload the image data
        media = drive_service.files().create(body=file_metadata, media_body=m, fields='id',
                                             media_mime_type="image/png").execute()

        # get the file ID and public URL of the uploaded image
        file_id = media.get('id')

        return f"https://drive.google.com/uc?export=view&id={file_id}"
    except Exception as e:
        print(f'An error occurred: {e}')
        raise


def generate_caption(background):
    prompt = f"generate a instagram caption for this prompt 'a beautiful woman at a {background} background.' it should be creative, " \
             f"cute and funny. Feel Good. Use Emojis. In first person. Also add relevant hashtags."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )

    return completion.choices[0].message["content"].replace('"', "").strip()


def generate_story_caption(background):
    prompt = f"generate a instagram story caption for the scene of {background} it should be creative, cute and funny. Feel Good. Use Emojis. In first person. Also add relevant hashtags. keep it only to few words"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )

    return completion.choices[0].message["content"].replace('"', "").strip()


def give_random_location_to_travel():
    res = requests.get("https://www.randomlists.com/data/world-cities-3.json")
    countries = json.loads(res.text)["RandL"]["items"]
    country = random.choice(countries)
    return country["name"] + ", " + country["detail"]


def generate_multiple_prompts(location):
    prompt = f"""Fashion and lifestyle influencer traveling to {location}, give me 4 list of places to go wearing different 
    stylish clothes to wear, describe it in details. describe background in details. as a prompt you give to stable 
    diffusion,  describe the background, scene in as much details as you can, use the following format "Place Name: ... 
    Description: ..." """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.9,
        messages=[
            {"role": "system",
             "content": "You are instagram influencer and prompt engineer. Respond in third person in "
                        "the list which was asked, nothing else."},
            {"role": "user", "content": prompt},
        ]
    )

    text = completion.choices[0].message["content"]
    final_prompts = []
    starting_prompt = "a beautiful and cute aashvi-500, single girl,"
    ending_prompt = "long haircut, light skin, (high detailed skin:1.3), 8k UHD DSLR, bokeh effect, soft lighting, " \
                    "high quality"
    index = 0
    sub_prompt = []
    print(text)
    location = ""
    for prompt in text.split("\n"):
        try:
            if prompt == "":
                continue

            is_place_name = False
            if not ("Place Name" in prompt or "Description" in prompt):
                continue
            is_place_name = "Place Name" in prompt
            # prompt = prompt[prompt.find(".") + 1:].strip()
            _, prompt = prompt.split(":")
            print(prompt)
            if is_place_name:
                index += 1
                location = prompt
                continue
            print(is_place_name, index)
            prompt = f"{starting_prompt} at {location}, {prompt}, {ending_prompt}"
            final_prompts.append([location, prompt])
            index += 1
            if index >= 1:
                pass
            #     index = 0
            #     print(is_place_name, index, sub_prompt)
            # final_prompts.append(sub_prompt)
            # sub_prompt = []
        except Exception as e:
            print(e)
            continue

    return final_prompts


def generate_multiple_story_prompts(location):
    prompt = f"""Give me prompts describing the beauty of {location}. Doing different activity, Be very descriptive for background."""
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.71,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "You are instagram influencer and prompt engineer"},
            {"role": "user", "content": prompt},
        ]
    )

    text = completion.choices[0].message["content"]

    final_prompts = []
    for prompt in text.split("\n"):
        if prompt == "":
            continue
        prompt = prompt[prompt.find(".") + 1:].strip()
        final_prompts.append(prompt)

    return final_prompts


# def generate_more_locations(how_many):
#     prompt = f"""this is my prompt "a portrait photo of beautiful Aashvi-500,standing in a modern art museum"
# Just respond with the imaginary scene location, random scene for instagram in the list format. no explanation just give me {how_many} of this."""
#
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         temperature=0.3,
#         frequency_penalty=0.8,
#         messages=[
#             {"role": "user", "content": prompt},
#         ]
#     )
#     points = completion.choices[0].message["content"].replace('"', "").replace("'", "")
#
#     # read the above response and generate a list of prompts
#     clean_points = []
#     for point in points.split("\n"):
#         clean_points.append(point.replace("\n", "")[point.find(".") + 1:].strip())
#
#     return clean_points


# def generate_more_rows(last_index, how_many=50):
#     negative_prompt = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"""
#     locations = generate_more_locations(how_many)
#     rows = []
#
#     for location in locations:
#         positive_prompt = f"""a portrait photo of beautiful aashvi-500,{location}, single person, smiling, white skin, cloudy eyes, thick long haircut, light skin, (high detailed skin:1.3), 8k UHD DSLR, bokeh effect, soft lighting, high quality"""
#         print(f"A{last_index + 2}:D{last_index + 2}")
#         rows.append({
#             "range": f"A{last_index + 2}:D{last_index + 2}",
#             "values": [[last_index, positive_prompt, negative_prompt, location, ]]
#         })
#         last_index += 1
#
#     # add more rows at the end the sheet
#     sheet.batch_update(rows, value_input_option="USER_ENTERED")


def check_if_automatic1111_is_active():
    global AUTOMATIC1111_URL
    with open("/Users/rikenshah/Desktop/Fun/insta-model/automatic1111_url.txt", "r") as f:
        AUTOMATIC1111_URL = f.read().strip()
    if AUTOMATIC1111_URL != "":
        print("URL:", AUTOMATIC1111_URL + "sdapi/v1/memory")
        response = requests.get(AUTOMATIC1111_URL + "sdapi/v1/memory")
        if response.status_code == 200:
            print("Automatic1111 is active")
            return AUTOMATIC1111_URL
    print("Automatic1111 is not active")
    return False


def close_automatic1111():
    global AUTOMATIC1111_URL, driver
    if driver is not None:
        driver.quit()
        driver = None
    AUTOMATIC1111_URL = ""


def get_driver():
    global driver
    options = uc.ChromeOptions()

    # uc.TARGET_VERSION = 120

    # or specify your own chromedriver binary (why you would need this, i don't know)

    # uc.install(
    #     executable_path='/Users/rikenshah/Desktop/Fun/insta-model/chromedriver.sh',
    # )

    driver = uc.Chrome(
        options=options, user_data_dir="/Users/rikenshah/Desktop/Fun/insta-model/profile-2",
    )
    return driver


def connect_to_automatic1111_api(user=0):
    global driver, AUTOMATIC1111_URL, options

    def get_automatic1111_url():
        global AUTOMATIC1111_URL, options
        retry = 15
        while retry != 0:
            try:
                found = False
                a = None
                for a in driver.find_elements(By.XPATH, "//colab-static-output-renderer/div[1]/div/pre/a"):
                    print(a.get_attribute("href"))
                    if a.get_attribute(
                            "href").endswith("trycloudflare.com/") or a.get_attribute("href").endswith(
                        "ngrok-free.app") or a.get_attribute("href").endswith("ngrok-free.app/"):
                        found = True
                        break
                if not found:
                    raise NoSuchElementException("Couldn't find the link")
                AUTOMATIC1111_URL = a.get_attribute("href")
                print("Found the link " + AUTOMATIC1111_URL)
                break
            except NoSuchElementException:
                print("Couldn't found the link, retrying in 10 secs...")
                pass
            except Exception as e:
                print(f"An error occurred: {e}")
            sleep(10)

            retry -= 1

    if check_if_automatic1111_is_active():
        return
    AUTOMATIC1111_URL = ""
    if driver is not None:
        driver.quit()
        # driver.close()
        options = uc.ChromeOptions()
    driver = get_driver()
    driver.get(f"https://colab.research.google.com/drive/1Ezb1humDZNX35w0YJHWbk7E7z-qA9WrX?authuser={user}")
    sleep(30)
    # Send CMD + M + . to open the console
    try:
        driver.find_element(By.CSS_SELECTOR, ".cell.running")
        print("Runtime is already running...")
        AUTOMATIC1111_URL = ""
        get_automatic1111_url()
        if check_if_automatic1111_is_active():
            return

        action = ActionChains(driver).key_down(Keys.COMMAND).send_keys("m").send_keys(".").key_up(Keys.COMMAND)
        action.perform()
        sleep(10)

        try:
            driver.find_element(By.XPATH, "//mwc-button[text()='Yes']").click()
            sleep(10)
            print("Stopping the runtime...")
        except NoSuchElementException:
            pass
    except NoSuchElementException:
        pass

    # Send CMD + F9 to open the console
    driver.find_element(By.TAG_NAME, "body").click()
    sleep(4)
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.COMMAND + Keys.F9)
    print("Waiting for the runtime to start...")
    sleep(5)
    try:
        driver.find_element(By.XPATH, "//mwc-button[text()='Run anyway']").click()
        sleep(10)
    except NoSuchElementException:
        pass

    print("Running Automatic1111...")
    sleep(20)
    try:
        driver.find_element(By.XPATH, "//mwc-button[text()='Connect without GPU']")
        close_automatic1111()
        connect_to_automatic1111_api(user + 1)
        return
    except NoSuchElementException:
        pass

    sleep(80)

    get_automatic1111_url()

    if not check_if_automatic1111_is_active():
        print("Couldn't find the link, please try again")
        driver.quit()
        driver = None
        exit(1)
        return None


def get_payload(type, prompt, seed, negative_prompt):
    payload = {}
    if type == "posts":
        payload = {
            "prompt": prompt,
            # "enable_hr": True,
            # "hr_resize_x": 800,
            # "hr_resize_y": 800,
            # "hr_upscaler": "R-ESRGAN 4x+",
            # "denoising_strength": 0.8,
            # "hr_second_pass_steps": 20,
            "seed": seed,
            "sampler_name": "DPM++ 2M",
            "batch_size": 1,
            "n_iter": 1,
            "steps": 120,
            "cfg_scale": 3.5,
            "width": 512,
            "height": 512,
            "restore_faces": True,
            "negative_prompt": negative_prompt,
            "send_images": True,
            "save_images": False,
        }
    elif type == "story":
        payload = {
            "prompt": prompt,
            "enable_hr": False,
            # "hr_resize_x": 1080,
            # "hr_resize_y": 1080,
            # "hr_upscaler": "R-ESRGAN 4x+",
            # "denoising_strength": 0.8,
            # "hr_second_pass_steps": 20,
            "seed": seed,
            "sampler_name": "DPM++ 2M Karras",
            "batch_size": 1,
            "n_iter": 1,
            "steps": 100,
            "cfg_scale": 7,
            "width": 720,
            "height": 1080,
            "restore_faces": True,
            "negative_prompt": negative_prompt,
            "send_images": True,
            "save_images": False,
        }
    return payload


def generate_posts_and_caption(sheet):
    content = sheet.get_all_records(value_render_option="FORMULA")
    image_cell = sheet.find('image')
    caption_cell = sheet.find('caption')
    generated_on = sheet.find('generated_on')
    hyperlink = sheet.find('hyperlink_image')
    negative_prompt = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, 
    anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, 
    morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, 
    blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, 
    malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck """
    for row in content:
        try:
            if row['prompt'] != '':
                if row['image'] == '':
                    connect_to_automatic1111_api()
                    payload = get_payload(row['type'], row['prompt'], row['seed'] if row["seed"] != "" else -1,
                                          row['negative_prompt'] if 'negative_prompt' in row else negative_prompt)
                    headers = {'Content-Type': 'application/json'}

                    print(f"Running for {row['prompt']} and location {row['location']}")
                    response = requests.post(f'{AUTOMATIC1111_URL}sdapi/v1/txt2img', headers=headers,
                                             data=json.dumps(payload), timeout=900)
                    if response.status_code != 200:
                        print("Failed to generate image with following error", response.json())
                        close_automatic1111()
                        exit()
                    # decode the image data and upload it to the sheet
                    img_data = response.json()['images'][0]
                    print("Image generated successfully")
                    # img_data = ""
                    index = row["index"] + 1
                    # upload image to Google Drive
                    image_url = upload_image_to_drive(f"{index}-aashvi.png", img_data,
                                                      '1rEysVX6M0vEZFYGbdDVc96G4ZBXYhDDs')
                    sheet.update_cell(image_cell.row + index, image_cell.col, f'=IMAGE("{image_url}", 4, 120, 120)')
                    # sheet.update_cell(image_cell.row + index, image_cell.col, image_url)
                    sheet.update_cell(generated_on.row + index, generated_on.col,
                                      datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    # image hyperlink
                    sheet.update_cell(hyperlink.row + index, hyperlink.col, f'=HYPERLINK("{image_url}", "Link")')

                if row['caption'] == '' and row['type'] == 'posts':
                    # generate the caption
                    try:
                        if row['type'] == "posts":
                            caption = generate_caption(row['location'])
                        else:
                            caption = generate_story_caption(row['prompt'])
                        sheet.update_cell(caption_cell.row + int(row["index"]) + 1, caption_cell.col, caption)
                    except Exception as e:
                        print(e)
                        continue
        except Exception as e:
            print("something went wrong", e)
            send_messages(f"Something went wrong while generating image {e}")
            continue
    close_automatic1111()


def get_last_index():
    if len(gsheet.get_all_values()) == 0:
        return -1
    return len(gsheet.get_all_records())


def generate_random_seed():
    return -1  # random.randint(0, 300000)


columns = ["index", "type", "prompt", "location", "group_id", "seed", "image", "generated_on", "caption", "approved",
           "posted_on_instagram", "hyperlink_image"]


def non_posted_instagram_posts():
    values = gsheet.get_all_records(value_render_option="FORMULA")
    group_ids = set()
    for row in values:
        if row["type"] == "posts" and row["posted_on_instagram"] == "":
            print(row["group_id"], row["index"])
            group_ids.add(row["group_id"])
    return len(group_ids)


def approved_non_posted_instagram_posts():
    values = gsheet.get_all_records(value_render_option="FORMULA")
    group_ids = set()
    for row in values:
        if row["type"] == "posts" and row["posted_on_instagram"] == "" and row["approved"] == "" and row["image"] != "":
            group_ids.add(row["group_id"])
    return len(group_ids)


def non_posted_story():
    values = gsheet.get_all_records(value_render_option="FORMULA")
    total = 0
    for row in values:
        if row["type"] == "story" and row["posted_on_instagram"] == "":
            total += 1
    return total


def approved_non_posted_story():
    values = gsheet.get_all_records(value_render_option="FORMULA")
    total = 0
    for row in values:
        if row["type"] == "story" and row["posted_on_instagram"] == "" and row["image"] != "" and row["approved"] == "":
            total += 1
    return total


def v1_generate_prompts():
    # if non_posted_instagram_posts() >= 5:
    #     print("There are more than 5 non posted instagram posts. Please post them first")
    #     return

    last_index = get_last_index()
    if last_index == -1:
        last_index = 0
        gsheet.insert_row(columns)

    values = []
    location = get_location()
    prompts = generate_multiple_prompts(location)
    group_id = generate_uuid()
    offset = last_index
    seed = generate_random_seed()
    first_caption = False
    caption = ""
    for prompt in prompts:
        values.append(
            [offset, "posts", prompt[1], prompt[0] + "," + location, group_id, seed, "", "", caption, "", "", ""])
        offset += 1
        if not first_caption:
            caption = "-"
            first_caption = True
    gsheet.insert_rows(values, row=last_index + 2)


def get_location():
    with open("/Users/rikenshah/Desktop/Fun/insta-model/location.txt", "r") as f:
        return f.read().strip()


def new_location(location):
    with open("/Users/rikenshah/Desktop/Fun/insta-model/location.txt", "w") as f:
        f.write(location)


def v1_generate_story_idea():
    # if non_posted_story() >= 5:
    #     print("There are more than 5 non posted story. Please post them first")
    #     return
    last_index = get_last_index()
    if last_index == -1:
        last_index = 0
        gsheet.insert_row(columns)

    values = []
    prompts = generate_multiple_story_prompts(get_location())
    offset = last_index
    for prompt in prompts:
        values.append([offset, "story", prompt, "", "", -1, "", "", "", "y", "", ""])
        offset += 1

    gsheet.insert_rows(values[:2], row=last_index + 2)


if __name__ == "__main__":
    # connect_to_automatic1111_api()
    # print(approved_non_posted_instagram_posts())
    # v1_generate_prompts()
    # v1_generate_story_idea()
    is_running_path = "/Users/rikenshah/Desktop/Fun/insta-model/is_running.txt"
    with open(is_running_path, "r") as f:
        is_running = f.read().strip()
        if is_running == "yep":
            print("Already running")
            exit()
    send_messages("Running image and caption generation...")

    with open(is_running_path, "w") as f:
        f.write("yep")
    generate_posts_and_caption(gsheet)
    send_messages("Stopping image and caption generation")
    with open(is_running_path, "w") as f:
        f.write("nope")
