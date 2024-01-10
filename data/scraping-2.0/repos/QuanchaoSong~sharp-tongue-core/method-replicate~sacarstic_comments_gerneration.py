import openai
from nltk.corpus import wordnet
from material_by_replicate import *
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import backoff


OPENAI_API_KEY = "Your OpenAI API Key"

class Sacarstic_Comments_Gerneration:
    def __init__(self) -> None:
        super().__init__()

        self.k = 5

        self.blip_tool = Material_By_Blip(k=self.k)
        # self.blip_tool = None

        openai.api_key = OPENAI_API_KEY

    def analyse_image_url(self, image_url):
        (self.context_sentence, self.element_list) = self.blip_tool.obtain_materials_from_image_url(image_url, count=self.k)
        # (self.context_sentence, self.element_list) = ("The Li River is so beautiful!", ["river", "mountain", "sky", "apple", "chair"])
        the_paraphrased_sentences = self.__generate_sacarstic_comments_to_elements()
        sacarstic_comment_list = self.__generate_sacarstic_comment_to_sentence(self.context_sentence)
        return (self.element_list, the_paraphrased_sentences, sacarstic_comment_list)
    
    def analyse_image_data(self, image_data):
        (self.context_sentence, self.element_list) = self.blip_tool.obtain_materials_from_image_data(image_data, count=self.k)
        the_paraphrased_sentences = self.__generate_sacarstic_comments_to_elements()
        sacarstic_comment_list = self.__generate_sacarstic_comment_to_sentence(self.context_sentence)
        return (self.element_list, the_paraphrased_sentences, sacarstic_comment_list)
    

    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    def __generate_sacarstic_comment_to_sentence(self, sentence):
        prompt = f"From different perspectives, generate 3 sarcastic comments towards the content of a picture: \"{sentence}\". It's better to give them a sarcastic tone. Give result in pure Python list like [\"comment1\", \"comment2\", \"comment3\"], adding escape mark if necessary."
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
        comment_list_str = answer_item["text"].strip()
        if (comment_list_str.lower().startswith("result:")):
            comment_list_str = comment_list_str[len("result:"):].strip()
        comment_list = eval(comment_list_str)
        return comment_list       

    def __generate_sacarstic_comments_to_elements(self):
        self.the_whole_adj_and_antonym_words_list = self.__get_adj_and_antonym_words_for_all_elements()
        print("the_whole_adj_and_antonym_words_list:", self.the_whole_adj_and_antonym_words_list)
        self.the_opposite_analogies = self.__get_opposite_analogies()
        print("the_opposite_analogies:", self.the_opposite_analogies)
        self.the_seed_association_sentences = self.__build_seed_association_sentences()
        print("the_seed_association_sentences:", self.the_seed_association_sentences)
        self.the_paraphrased_all_sentences = self.__paraphrase_all_sentences()
        print("the_paraphrased_all_sentences:", self.the_paraphrased_all_sentences)
        return self.the_paraphrased_all_sentences
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def __get_adj_and_antonym_words_for_all_elements(self):
        print("\n==================Finding adjectives & antonyms==================\n")
        whole_list = []
        prompt_list = []
        for ele in self.element_list:
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
        res = f"List {self.k} non-negative mostly-used adjective words and their corresponding antonyms regarding the object: \"{element}\". Give result in pure python 2-d array form like [[adj, antonym], [adj, antonym], ...], without nonsense like \"Answer:\" or \"Answer=\"."
        return res
    
    def __get_opposite_analogies(self):
        print("\n==================Finding analogies(opposite ones)==================")
        res = []
        for i in range(len(self.the_whole_adj_and_antonym_words_list)):
            sub_adj_and_antonym_words_lst = self.the_whole_adj_and_antonym_words_list[i]
            ele = self.element_list[i]
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
    # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
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
        for i in range(len(self.element_list)):
            sub_res = []
            ele = self.element_list[i]
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
            ele = self.element_list[i]
            seed_association_sentence_sub_list = self.the_seed_association_sentences[i]
            for j in range(len(seed_association_sentence_sub_list)):                
                seed_association_sentence = seed_association_sentence_sub_list[j]
                paraphrase_single_sentence_prompt = self.__generate_prompt_to_paraphrase_single_sentence(seed_association_sentence)
                prompt_list.append(paraphrase_single_sentence_prompt)
            paraphrased_sentences_to_ele = self.__get_paraphrased_sentences_to_element(ele, prompt_list)
            res.append(paraphrased_sentences_to_ele)
        return res
    
    @retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6))
    # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
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
    tool_for_elements = Sacarstic_Comments_Gerneration()
    res = tool_for_elements.analyse_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    print("res:", res)