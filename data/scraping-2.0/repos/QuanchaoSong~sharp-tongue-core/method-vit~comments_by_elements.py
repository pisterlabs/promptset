from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import requests
import random
import openai
from nltk.corpus import wordnet
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


OPENAI_API_KEY = "Your OpenAI API Key"

class Comments_By_Elements:
    def __init__(self) -> None:
        super().__init__()

        self.k = 5

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        openai.api_key = OPENAI_API_KEY
    
    def analyse_image_url(self, image_url):
        the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def analyse_image_local_path(self, image_local_path):
        the_image = Image.open(image_local_path).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def analyse_image_data(self, image_data):
        the_image = Image.open(image_data).convert("RGB")
        return self.__analyse_image(image=the_image)
    
    def __analyse_image(self, image):
        element_list = self.__get_elements_by_vit(image, k=self.k)
        print("element_list:", element_list)
        purged_element_list = self.__purge_elements(element_list)
        print("purged_element_list:", purged_element_list)
        self.the_more_common_names = self.__get_more_common_names(purged_element_list)
        print("the_more_common_names:", self.the_more_common_names)
        self.the_whole_adj_and_antonym_words_list = self.__get_adj_and_antonym_words_for_all_elements()
        print("the_whole_adj_and_antonym_words_list:", self.the_whole_adj_and_antonym_words_list)
        self.the_opposite_analogies = self.__get_opposite_analogies()
        print("the_opposite_analogies:", self.the_opposite_analogies)
        self.the_seed_association_sentences = self.__build_seed_association_sentences()
        print("the_seed_association_sentences:", self.the_seed_association_sentences)
        self.the_paraphrased_all_sentences = self.__paraphrase_all_sentences()
        print("the_paraphrased_all_sentences:", self.the_paraphrased_all_sentences)
        return (self.the_more_common_names, self.the_paraphrased_all_sentences)

    def __get_elements_by_vit(self, image, k=5):
        print("\n==================Extracting objects==================")
        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()
        # print("logits:", logits)
        elements_res = []
        # model predicts one of the 1000 ImageNet classes
        vs, ds = torch.topk(logits, k)
        # print("vs:", vs)
        for i in range(ds.shape[0]):
            d = ds[i]
            predicted_label = self.model.config.id2label[d.item()]
            # print("%.2f, %s" % (vs[i].item(), predicted_label))
            elements_res.append(predicted_label)

        return elements_res
    
    def __purge_elements(self, lst):
        res = []
        for i in range(len(lst)):
            ele = lst[i]
            seperator = ","
            if (seperator in ele):
                parts = ele.split(seperator)
                rdm_part = random.choice(parts).strip()
                res.append(rdm_part)
            else:
                res.append(ele)
        return res
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __get_more_common_names(self, lst):
        print("\n==================Finding commonly-used names==================")
        prompt = f"Find other commonly-used names for each of these objects: \"{lst}\". Give result in a pure Python array."
        # print("__get_more_common_names prompt:", prompt)

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("__get_more_common_names response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        name_list_str = answer_item["text"].strip().lower()
        if (name_list_str.startswith("answer:")):
            name_list_str = name_list_str[len("answer:"):]
        name_list = eval(name_list_str)
        return name_list        


    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __get_adj_and_antonym_words_for_all_elements(self):
        print("\n==================Finding adjectives & antonyms==================\n")
        whole_list = []
        prompt_list = []
        for ele in self.the_more_common_names:
            prompt = self.__generate_adj_words_prompt_for_element(ele)
            prompt_list.append(prompt)
        
        # print("prompt_list:", prompt_list)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_list,
            temperature=0.7,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("__get_adj_and_antonym_words_for_all_elements response:", response)

        choices = response["choices"]
        for answer_item in choices:
            words_list_str = answer_item["text"].strip().lower()
            if (words_list_str.startswith("answer:")):
                words_list_str = words_list_str[len("answer:"):]
            # print("words_list_str:", words_list_str)
            adj_and_antonym_words_list = eval(words_list_str)
            whole_list.append(adj_and_antonym_words_list)
        return whole_list

    def __generate_adj_words_prompt_for_element(self, element):
        res=f"List {self.k} non-negative mostly-used adjective words and their corresponding antonyms regarding the object: \"{element}\". Give result in pure python 2-d array form, without nonsense like \"Answer:\"."
        return res
    
    def __get_opposite_analogies(self):
        print("\n==================Finding analogies(opposite ones)==================")
        res = []
        for i in range(len(self.the_whole_adj_and_antonym_words_list)):
            sub_adj_and_antonym_words_lst = self.the_whole_adj_and_antonym_words_list[i]
            ele = self.the_more_common_names[i]
            prompt_list = []
            for j in range(len(sub_adj_and_antonym_words_lst)):                
                adj_words_pair = sub_adj_and_antonym_words_lst[j]
                antonym_word = adj_words_pair[1]
                opposite_analogy_prompt = self.__generate_prompt_to_get_opposite_analogy_to_element(ele, antonym_word)
                prompt_list.append(opposite_analogy_prompt)
            
            opposite_analogies_to_ele = self. __get_opposite_analogies_to_element(ele, prompt_list)
            res.append(opposite_analogies_to_ele)
        
        return res
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __get_opposite_analogies_to_element(self, ele, prompt_list):
        print(f"================finding opposite analogy to {ele}===============")
        # time.sleep(10)
        res = []
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_list,
            temperature=0.7,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print(f"__get_opposite_analogies_to_element: {ele}, response: {response}")
        choices = response["choices"]
        for answer_item in choices:
            opposite_analogy = answer_item["text"].strip().lower()
            if (opposite_analogy.endswith(".")):
                opposite_analogy = opposite_analogy[:-1]
            res.append(opposite_analogy)

        return res
    
    def __generate_prompt_to_get_opposite_analogy_to_element(self, element, antonym):
        opposite_term = antonym + " " + element
        prompt = f"Like that \"A snail in a swimming pool\" is an analogy to \"slow boat\", what is the thing that can be used as an anology to \"{opposite_term}\"? Note that the analogy has to be of different category from the \"{opposite_term}\". Just give the result, without nonsense."
        return prompt
    
    def __build_seed_association_sentences(self):
        print("\n==================Building associations for scenes & analogies==================")
        res = []
        for i in range(len(self.the_more_common_names)):
            sub_res = []
            ele = self.the_more_common_names[i]
            adj_antonym_pair_sub_lst = self.the_whole_adj_and_antonym_words_list[i]
            opposite_analogy_sub_list = self.the_opposite_analogies[i]
            for j in range(len(adj_antonym_pair_sub_lst)):
                adj_antonym_pair = adj_antonym_pair_sub_lst[j]
                adj_word = adj_antonym_pair[0]
                opposite_analogy = opposite_analogy_sub_list[j]
                seed_association_sentence = f"such a {adj_word} {ele}, like {opposite_analogy}"
                sub_res.append(seed_association_sentence)
            res.append(sub_res)
        return res
    
    def __build_seed_association_sentences(self):
        print("\n==================Building associations for scenes & analogies==================")
        res = []
        for i in range(len(self.the_more_common_names)):
            sub_res = []
            ele = self.the_more_common_names[i]
            adj_antonym_pair_sub_lst = self.the_whole_adj_and_antonym_words_list[i]
            opposite_analogy_sub_list = self.the_opposite_analogies[i]
            for j in range(len(adj_antonym_pair_sub_lst)):
                adj_antonym_pair = adj_antonym_pair_sub_lst[j]
                adj_word = adj_antonym_pair[0]
                opposite_analogy = opposite_analogy_sub_list[j]
                seed_association_sentence = f"such a {adj_word} {ele}, like {opposite_analogy}"
                sub_res.append(seed_association_sentence)
            res.append(sub_res)
        return res
    
    def __paraphrase_all_sentences(self):
        print("\n==================Paraphrasing==================")
        res = []
        for i in range(len(self.the_seed_association_sentences)):
            prompt_list = []
            ele = self.the_more_common_names[i]
            seed_association_sentence_sub_list = self.the_seed_association_sentences[i]
            for j in range(len(seed_association_sentence_sub_list)):                
                seed_association_sentence = seed_association_sentence_sub_list[j]
                paraphrase_single_sentence_prompt = self.__generate_prompt_to_paraphrase_single_sentence(seed_association_sentence)
                prompt_list.append(paraphrase_single_sentence_prompt)
            paraphrased_sentences_to_ele = self.__get_paraphrased_sentences_to_element(ele, prompt_list)
            res.append(paraphrased_sentences_to_ele)
        return res
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __get_paraphrased_sentences_to_element(self, ele, prompt_list):
        print(f"================paraphrasing for {ele}===============")
        # time.sleep(20)
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

        # print(f"__get_paraphrased_sentences_to_element: {ele}, response: {response}")
        choices = response["choices"]
        for answer_item in choices:
            paraphrased_sentence = answer_item["text"].strip()
            res.append(paraphrased_sentence)
        return res
    
    def __generate_prompt_to_paraphrase_single_sentence(self, single_sentence):
        prompt = f"Paraphrase and extend(if necessary) sentence \"{single_sentence}\", to make it as vivid as if it was from a real person. Specifically, it is better to give it a sarcastic tone."
        return prompt

if __name__ == '__main__':
    tool_for_elements = Comments_By_Elements()
    res = tool_for_elements.analyse_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    # res = tool_for_elements.analyse_image_local_path("/Users/albus/Downloads/AI-works/AI-Test/Z-Images/Cafe.jpg")
    print("\nres:", res)