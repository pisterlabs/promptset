import clip
import cv2
import openai
from PIL import Image
import requests
import random
import numpy as np
from deepface import DeepFace
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

OPENAI_API_KEY = "Your OpenAI API Key"

class Comments_For_Face:
    def __init__(self) -> None:
        super().__init__()

        self.k = 5

        self.frontalface_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.frontalface_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        openai.api_key = OPENAI_API_KEY

    def analyse_image_url(self, image_url):
        the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        return self.__analyze_PIL_image(the_image)

    def analyse_image_data(self, image_data):
        the_image = Image.open(image_data).convert("RGB")
        return self.__analyze_PIL_image(the_image)

    def __analyze_PIL_image(self, PIL_image):
        (self.face_exist, self.cropped_face_image) = self.__check_face(PIL_image)
        if (self.face_exist == False):
            return (False, None, None)
        
        print("\n==============Face Analysis==============\n")
        self.face_basic_info = self.__get_face_emotion_and_other_properties(self.cropped_face_image)
        print("\nface_basic_info:", self.face_basic_info)
        (self.attractiveness_index, self.attractiveness_value) = self.__get_attractiveness_of_face(self.cropped_face_image, self.face_basic_info["dominant_gender"])
        print("\nattractiveness_value:", self.attractiveness_value)
        self.basic_scene = self.__generate_basic_scene()
        print("\nbasic_scene:", self.basic_scene)
        self.adj_and_antonym_pair_list = self.__get_adj_and_antonym_list()
        print("\nadj_and_antonym_pair_list:", self.adj_and_antonym_pair_list)
        self.opposite_analogy_list = self.__get_opposite_analogies()
        print("\nopposite_analogy_list:", self.opposite_analogy_list)
        self.seed_sentence_list = self.__build_association_list()
        print("\nseed_sentence_list:", self.seed_sentence_list)
        self.paraphrased_sentence_list = self.__paraphrase_sentences()
        print("\nparaphrased_sentence_list:", self.paraphrased_sentence_list)

        return (self.face_exist, self.basic_scene, self.paraphrased_sentence_list)

    def __insert_str(self, original_str, pos, adding_str):
        str_list = list(original_str)
        str_list.insert(pos, adding_str)
        res_str = ''.join(str_list)
        return  res_str

    def __reframe_box(self, box, image_size):
        (width, height) = image_size
        delta = 150
        (x, y, w, h) = box
        x = max(0, x-delta)
        y = max(0, y-delta)
        w = min(width, w+(2*delta))
        h = min(height, h+(2*delta))
        return (x, y, w, h)

    def __check_face(self, image):
        faces = self.face_cascade.detectMultiScale(np.array(image), scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
        if (len(faces) == 0):
            return (False, None)
        else:
            a_face_box = random.choice(faces)
            (x, y, w, h) = self.__reframe_box(a_face_box, image.size)
            cropped_image = image.crop((x, y, x+w, y+h))
            return (True, cropped_image)
        
    
    def __get_face_emotion_and_other_properties(self, image):
        face_analysis_list = DeepFace.analyze(img_path=np.array(image))
        # if (len(face_analysis_list) == 0):
        #     return None
        face_analysis = face_analysis_list[0]
        # print("face_analysis:", face_analysis)
        res_dic = {}
        res_dic["dominant_emotion"] = face_analysis["dominant_emotion"].lower()
        res_dic["dominant_gender"] = face_analysis["dominant_gender"].lower()
        res_dic["dominant_race"] = face_analysis["dominant_race"].lower()
        res_dic["age"] = face_analysis["age"]

        return res_dic
    
    def __get_attractiveness_of_face(self, face_image, gender):
        cls_names = ["beautiful", "ugly"]
        if (gender == "man"):
            cls_names = ["handsome", "ugly"]

        image = self.preprocess(face_image).unsqueeze(0).to(self.device)
        text = clip.tokenize(cls_names).to(self.device)
        with torch.no_grad():
            logits_per_image, _ = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            # print("probs:", probs)
            return (probs.argmax(), probs.max())
    
    def __generate_person_identity(self):
        age = self.face_basic_info["age"]
        gender = self.face_basic_info["dominant_gender"]

        res = ""
        if (age < 20):
            res += "young "
            if (gender == "man"):                
                res += "boy"
            else:
                res += "girl"
        else:
            if (age > 50):
                res += "old "
            if (gender == "man"):                
                res += "man"
            else:
                res += "woman"

        return res

    def __generate_basic_scene(self):
        age = self.face_basic_info["age"]
        gender = self.face_basic_info["dominant_gender"]
        basic_scene = ("a " + str(age) + "-year-old ")

        if (self.attractiveness_index == 0):
            face_adj = "beautiful"
            if (gender == "man"):
                face_adj = "handsome"

            if (self.attractiveness_value < 0.80):
                basic_scene += ("relatively " + face_adj + " ")
            else:
                basic_scene += (face_adj + " ")

        basic_scene += self.__generate_person_identity()
        return basic_scene
    
    def __get_adj_and_antonym_pair_prompt(self, scene):
        res = f"List {self.k} non-negative mostly-used adjective words and their corresponding antonyms towards: \"{scene}\". Give result in pure python 2-d array form like [[adj, antonym], [adj, antonym], ...], without nonsense like \"Answer:\" or \"Answer=\"."
        return res
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __get_adj_and_antonym_list(self):
        print(f"\n==================Finding adjectives & antonyms for \"{self.basic_scene}\"==================\n")
        prompt = self.__get_adj_and_antonym_pair_prompt(self.basic_scene)
        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        adj_and_antonym_pair_list_str = answer_item["text"].strip().upper()
        adj_and_antonym_pair_list = eval(adj_and_antonym_pair_list_str)
        return adj_and_antonym_pair_list
    
    def __generate_opposite_analogy_prompt(self, oppsite_scene):
        prompt = f"Like that \"A snail in a swimming pool\" is an analogy to \"slow boat\", what is the thing that can be used as an anology to \"{oppsite_scene}\"? Note that the analogy has to be of different category from the \"{oppsite_scene}\". Just give the result, without nonsense."
        return prompt
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __get_opposite_analogies(self):
        prompt_list = []
        for i in range(len(self.adj_and_antonym_pair_list)):
            adj_and_antonym_pair = self.adj_and_antonym_pair_list[i]
            adj_word = adj_and_antonym_pair[0]
            antonym_word = adj_and_antonym_pair[1]
            oppsite_scene = self.__insert_str(self.basic_scene, 2, (antonym_word + " "))
            prompt = self.__generate_opposite_analogy_prompt(oppsite_scene)
            prompt_list.append(prompt)
        # print("prompt_list:", prompt_list)

        res = []
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_list,
            temperature=1,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        for answer_item in choices:
            opposite_analogy = answer_item["text"].strip().lower()
            res.append(opposite_analogy)
        return res
    
    def __build_association_list(self):
        association_list = []
        for i in range(len(self.adj_and_antonym_pair_list)):
            adj_and_antonym_pair = self.adj_and_antonym_pair_list[i]
            adj_word = adj_and_antonym_pair[0]
            opposite_analogy = self.opposite_analogy_list[i]
            person_identity = self.__generate_person_identity()
            seed_sentence = f"such a {adj_word} {person_identity}, like {opposite_analogy}"
            # print("seed_sentence:", seed_sentence)
            association_list.append(seed_sentence)
        return association_list
    
    def __generate_paraphrase_prompt(self, single_sentence):
        prompt = f"Paraphrase and extend(if necessary) sentence \"{single_sentence}\", to make it as vivid as if it was from a real person. Specifically, it is better to give it a sarcastic tone."
        return prompt
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __paraphrase_sentences(self):
        prompt_list = []
        for i in range(len(self.seed_sentence_list)):
            seed_sentence = self.seed_sentence_list[i]
            # print("seed_sentence:", seed_sentence)
            prompt = self.__generate_paraphrase_prompt(seed_sentence)
            prompt_list.append(prompt)
        # print("prompt_list:", prompt_list)

        res = []
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_list,
            temperature=1,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        for answer_item in choices:
            opposite_analogy = answer_item["text"].strip().lower()
            res.append(opposite_analogy)
        return res


if __name__ == '__main__':
    tool_for_face = Comments_For_Face()
    res = tool_for_face.analyse_image_url("https://th.bing.com/th/id/OIP.yDPUz4c9NYXJrmv8FKaASwHaEK?pid=ImgDet&rs=1")
    print("res:", res)