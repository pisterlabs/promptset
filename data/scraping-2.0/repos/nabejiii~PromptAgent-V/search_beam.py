import os
from dotenv import load_dotenv
import csv
from openai import OpenAI
import queue 

from create_init_prompt import create_init_prompt
from improvement import improve_prompt, adjust_prompt_improvement
from image import image_val, create_image, save_image

# from mock import create_init_prompt, improve_prompt, image_val, create_image, save_image, adjust_prompt_improvement


class search_beam():
    def __init__(self, image_num, dir_name, beam_width, num):
        load_dotenv()
        self.image_num = image_num
        self.origin_image = os.path.join("data", "image_" + str(image_num), "origin_" + str(image_num) + ".jpg")
        if not os.path.exists(self.origin_image):
            raise Exception("The image does not exist: " + self.origin_image)
        
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY_V"],
        )
        self.directory = os.path.join("data", "image_" + str(self.image_num), dir_name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.num = num
        self.beam_width = beam_width
        self.prompts = []
        self.diffs = []
        self.images = []
        self.image_layers = []
        self.scores = []
        self.current_top_beams = queue.Queue() # current top beams
        
        init_prompt = create_init_prompt(self.origin_image)
        init_image_http = create_image(init_prompt)
        init_image = os.path.join(self.directory, "image_" + str(self.image_num) + "_" + str(len(self.images)) + ".jpg")
        save_image(init_image, init_image_http)
        
        self.prompts.append(init_prompt)
        self.diffs.append(0)
        self.images.append(init_image)
        init_layer = 0
        self.image_layers.append(init_layer)
        init_score = image_val(self.origin_image, init_image)
        self.scores.append(init_score)

        self.current_top_beams.put((init_prompt, init_image, init_layer, init_score, 0))

    def beam_step(self, layer):
        print(f"Beam step layer: {layer}")
        new_beam = []
        current_img_num = 0
        while not self.current_top_beams.empty():
            prompt, image, image_layer, score, diff = self.current_top_beams.get()
            for j in range(self.beam_width):
                print(f"Beam step layer: {layer}, image: {current_img_num}")
                # generate image
                diff, new_prompt = improve_prompt(self.origin_image, image, prompt)
                new_image_http = create_image(new_prompt)
                new_image = os.path.join(self.directory, "image_" + str(self.image_num) + "_" + str(image_layer + 1) + "_" + str(current_img_num) + ".jpg")
                save_image(new_image, new_image_http)
                new_score = image_val(self.origin_image, new_image)
                new_beam.append((new_prompt, new_image, image_layer + 1, new_score, diff))
                current_img_num += 1

        sorted_beam = sorted(new_beam, key=lambda x: x[3])

        beam_num = 0
        for beam in sorted_beam:
            if beam_num < self.num:
                self.current_top_beams.put(beam)
            self.prompts.append(beam[0])
            self.images.append(beam[1])
            self.image_layers.append(beam[2])
            self.scores.append(beam[3])
            self.diffs.append(beam[4])
            beam_num += 1

    def beam_step_with_learning_rate(self, layer, progress):
        print(f"Beam step layer: {layer}")
        new_beam = []
        current_img_num = 0
        while not self.current_top_beams.empty():
            prompt, image, image_layer, score, diff = self.current_top_beams.get()
            for j in range(self.beam_width):
                print(f"Beam step layer: {layer}, image: {current_img_num}")
                # generate image
                diff, new_prompt = adjust_prompt_improvement(self.origin_image, image, prompt, progress)
                new_image_http = create_image(new_prompt)
                new_image = os.path.join(self.directory, "image_" + str(self.image_num) + "_" + str(image_layer + 1) + "_" + str(current_img_num) + ".jpg")
                save_image(new_image, new_image_http)
                new_score = image_val(self.origin_image, new_image)
                new_beam.append((new_prompt, new_image, image_layer + 1, new_score, diff))
                current_img_num += 1

        sorted_beam = sorted(new_beam, key=lambda x: x[3])

        beam_num = 0
        for beam in sorted_beam:
            if beam_num < self.num:
                self.current_top_beams.put(beam)
            self.prompts.append(beam[0])
            self.images.append(beam[1])
            self.image_layers.append(beam[2])
            self.scores.append(beam[3])
            self.diffs.append(beam[4])
            beam_num += 1
        
    def search_beam(self, max_layer):
        for i in range(max_layer):
            self.beam_step(i)

        self.store_evaluation()

    def search_beam_with_learning_rate(self, max_layer):
        for i in range(max_layer):
            progress = i / max_layer
            self.beam_step_with_learning_rate(i, progress)

        self.store_evaluation()

    def store_evaluation(self):
        file_path = os.path.join(self.directory, "evaluation.csv")
        image_id = 0

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Prompt", "Diff", "Image", "Layer", "Evaluation"])
            for prompt, diff, image, layer, score in zip(self.prompts, self.diffs, self.images, self.image_layers, self.scores):
                writer.writerow([image_id, prompt, diff, image, layer, score])
                image_id += 1

        print(f"Prompts and evaluations successfully saved to {file_path}")

if __name__ == "__main__":
    search_beam = search_beam(8, "search_beam", 3, 2)
    # search_beam.search_beam(3) #max_layer
    search_beam.search_beam_with_learning_rate(3) #max_layer