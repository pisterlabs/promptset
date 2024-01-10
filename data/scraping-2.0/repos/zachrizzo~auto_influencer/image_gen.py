import os
import random
import base64
from PIL import Image
from io import BytesIO
from gradio_client import Client
import asyncio
import time

from selenium.webdriver.common.by import By
#import ActionChains and WebDriverWait and EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import requests
import datetime
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import openai
from selenium import webdriver

class ImageGenerator:
    def __init__(self):
        self.url = "http://127.0.0.1:7865/"
        self.client = Client(self.url, serialize=False)
        self.driver = None


    def _convert_img_to_base64(self, img_path):
        with Image.open(img_path) as image:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            print("Image size:", image.size)
            # print(base64.b64encode(buffered.getvalue()).decode())
            return base64.b64encode(buffered.getvalue()).decode()

    async def continuous_img_gen(self, prompt, negative_prompt, performance, number_of_photos, img_prompt, img_face_path, img_face_stop_at, img_face_weight, img_prompt_1_path, img_prompt_stop_at_1, img_prompt_weight_1):
        img_face_base64 = self._convert_img_to_base64(img_face_path)


        img_1_files = [os.path.join(img_prompt_1_path, file) for file in os.listdir(img_prompt_1_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_path in img_1_files:
            img_prompt_base64 = self._convert_img_to_base64(img_path)

            result =  await self.client.predict(
                prompt, negative_prompt, ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp'], performance, '1152×896 <span style="color: grey;"> ∣ 9:7</span>', number_of_photos, random.randint(0, 2**63 - 1), 2, 4, 'juggernautXL_version6Rundiffusion.safetensors', 'None', 0.5, 'sd_xl_offset_example-lora_1.0.safetensors', 0.1, 'None', 1, 'None', 1, 'None', 1, 'None', 1, img_prompt, 'uov', 'Disabled', None, [], None, '', f'data:image/png;base64,{img_face_base64}', img_face_stop_at, img_face_weight, 'FaceSwap', f'data:image/png;base64,{img_prompt_base64}', img_prompt_stop_at_1, img_prompt_weight_1, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt',
                fn_index=30
            )

            while result is None:
                time.sleep(1)

            print("Predict result type:", type(result))
            print("Predict result content:", result)





    def generate_image(self, prompt, image_path):
        # Convert the image to a Base64 encoded string
        image_base64 = self._convert_img_to_base64(image_path)

        # Prediction logic with the Base64 encoded image
        result = self.client.predict(
            prompt, '', ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp'], 'Extreme Speed', '1152×896 <span style="color: grey;"> ∣ 9:7</span>', 1, '3876750373260134344', 2, 4, 'juggernautXL_version6Rundiffusion.safetensors', 'None', 0.5, 'sd_xl_offset_example-lora_1.0.safetensors', 0.1, 'None', 1, 'None', 1, 'None', 1, 'None', 1, False, 'uov', 'Disabled', None, [], None, '', image_base64, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt',
            fn_index=30  # Function index for predict
        )

        return result



    def continuous_img_gen_using_ui(self,fullScreen, prompt, negative_prompt, performance, number_of_photos, img_prompt, img_face_path, img_face_stop_at, img_face_weight, img_prompt_1_path, img_prompt_stop_at_1, img_prompt_weight_1):
        self.driver = webdriver.Firefox()
        self.driver.get(self.url)
        if fullScreen:
            self.driver.maximize_window()
        time.sleep(5)

        prompt_input = self.driver.find_element(By.CSS_SELECTOR, '#positive_prompt > label > textarea')
        gen_button = self.driver.find_element(By.CSS_SELECTOR, '#generate_button')
        img_prompt_checkbox = self.driver.find_element(By.CSS_SELECTOR, '#component-16 > label > input')
        advanced_checkbox = self.driver.find_element(By.CSS_SELECTOR, '#component-17 > label > input')
        advanced_checkbox.click()
        time.sleep(1)

        num_of_imgs = self.driver.find_element(By.CSS_SELECTOR, '#component-88 > div.wrap.svelte-1cl284s > div > input')
        num_of_imgs.clear()
        num_of_imgs.send_keys(number_of_photos)
        time.sleep(.5)

        print(performance)

        if performance == 'Extreme Speed':
            extreme_speed = self.driver.find_element(By.CSS_SELECTOR, '#component-86 > div.wrap.svelte-1p9xokt > label.svelte-1p9xokt.selected > input')
            extreme_speed.click()
        elif performance == 'Speed':
            speed = self.driver.find_element(By.CSS_SELECTOR, '#component-86 > div.wrap.svelte-1p9xokt > label:nth-child(1) > input')
            speed.click()
        elif performance == 'Quality':
            quality = self.driver.find_element(By.CSS_SELECTOR, '#component-86 > div.wrap.svelte-1p9xokt > label:nth-child(2) > input')
            quality.click()

        if prompt != '':
            prompt_input.send_keys(prompt)
            time.sleep(.5)

        if img_prompt:
            img_prompt_checkbox.click()
            time.sleep(2)

            img_prompt_tab = self.driver.find_element(By.CSS_SELECTOR, '#component-19 > div:nth-child(1) > button:nth-child(2)')
            img_prompt_tab.click()
            time.sleep(2)

            print('clicked')

            #scroll to and click advanced
            img_advanced = self.driver.find_element(By.CSS_SELECTOR, '#component-62 > label > input')
            self.driver.execute_script("arguments[0].scrollIntoView();", img_advanced)
            img_advanced.click()
            time.sleep(1)


            if img_face_path != '':

                #scroll to and click face swap
                img_face_swap = self.driver.find_element(By.CSS_SELECTOR, '#component-37 > div:nth-child(3) > label:nth-child(4) > input:nth-child(1)')
                self.driver.execute_script("arguments[0].scrollIntoView();", img_face_swap)
                img_face_swap.click()
                time.sleep(.5)


                #scroll to and upload face image
                img_face = self.driver.find_element(By.CSS_SELECTOR, '#component-31 > div.image-container.svelte-p3y7hu > div > input')
                self.driver.execute_script("arguments[0].scrollIntoView();", img_face)
                img_face.send_keys(img_face_path)
                time.sleep(1)


                #upload face image
                img_face_stop_at_slider = self.driver.find_element(By.CSS_SELECTOR, '#component-34 > div.wrap.svelte-1cl284s > div > input')
                img_face_stop_at_slider.clear()
                img_face_stop_at_slider.send_keys(img_face_stop_at)
                time.sleep(.5)

                img_face_weight_slider = self.driver.find_element(By.CSS_SELECTOR, '#component-35 > div.wrap.svelte-1cl284s > div > input')
                img_face_weight_slider.clear()
                img_face_weight_slider.send_keys(img_face_weight)
                time.sleep(.5)

            if img_prompt_1_path != '':
                img_1_files = [os.path.join(img_prompt_1_path, file) for file in os.listdir(img_prompt_1_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_path in img_1_files:
                    #scroll to and upload image 1
                    img_1 = self.driver.find_element(By.CSS_SELECTOR, '#component-39 > div.image-container.svelte-p3y7hu > div > input')
                    self.driver.execute_script("arguments[0].scrollIntoView();", img_1)
                    img_1.send_keys(img_path)
                    time.sleep(1)

                    #upload image 1
                    img_1_stop_at_slider = self.driver.find_element(By.CSS_SELECTOR, '#component-42 > div.wrap.svelte-1cl284s > div > input')
                    img_1_stop_at_slider.clear()
                    img_1_stop_at_slider.send_keys(img_prompt_stop_at_1)
                    time.sleep(.5)

                    img_1_weight_slider = self.driver.find_element(By.CSS_SELECTOR, '#component-43 > div.wrap.svelte-1cl284s > div > input')
                    img_1_weight_slider.clear()
                    img_1_weight_slider.send_keys(img_prompt_weight_1)
                    time.sleep(5)

                    while True:
                        gen_button = self.driver.find_element(By.CSS_SELECTOR, '#generate_button')
                        if gen_button.is_displayed():
                            gen_button.click()
                            time.sleep(3)
                            break










# Example usage
# generator = ImageGenerator()
# generator.generate_image('a cat on the moon')
