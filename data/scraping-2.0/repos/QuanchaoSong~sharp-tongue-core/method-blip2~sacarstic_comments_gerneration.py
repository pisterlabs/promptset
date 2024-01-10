from transformers import AutoImageProcessor, ViTForImageClassification
import torch
import openai
from nltk.corpus import wordnet
from material_by_blip import *


OPENAI_API_KEY = "Your OpenAI API Key"

class Sacarstic_Comments_Gerneration:
    def __init__(self) -> None:
        super().__init__()

        self.blip_tool = Material_By_Blip()
        
        openai.api_key = OPENAI_API_KEY

    def analyse_image_url(self, image_url):
        (context_sentence, element_list) = self.blip_tool.obtain_materials_from_image_url(image_url)
        the_paraphrased_sentences = self.__generate_sacarstic_comments_to_elements(element_list)
        sacarstic_comment = self.__generate_sacarstic_comment_to_sentence(context_sentence)
        return (element_list, the_paraphrased_sentences, sacarstic_comment)
    
    def analyse_image_data(self, image_data):
        (context_sentence, element_list) = self.blip_tool.obtain_materials_from_image_data(image_data)
        the_paraphrased_sentences = self.__generate_sacarstic_comments_to_elements(element_list)
        sacarstic_comment = self.__generate_sacarstic_comment_to_sentence(context_sentence)
        return (element_list, the_paraphrased_sentences, sacarstic_comment)
    
    def __generate_sacarstic_comment_to_sentence(self, sentence):
        prompt = f"Generate a sarcastic comment to this sentence:\"{sentence}\"."
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        comment = answer_item["text"].strip()
        return comment        

    def __generate_sacarstic_comments_to_elements(self, element_list):
        the_adj_words_list = self.__get_adj_words(element_list)
        the_antonym_word_list = self.__get_antonym_word_list(the_adj_words_list)
        the_combined_antonyms_with_items = self.__combined_antonyms_with_items(the_antonym_word_list, element_list)
        the_anology_list = self.__get_analogies_by_openai(the_combined_antonyms_with_items)
        the_seed_sentence_list = self.__create_seed_sentence_list(the_anology_list, the_adj_words_list, element_list)
        the_paraphrased_sentences = self.__paraphrase_sentences(the_seed_sentence_list)
        return the_paraphrased_sentences

    def __generate_adj_words_prompt(self, lst):
        res = f"List 5 non-negative adjective words for each of these noun words respectively: {lst}. Give result in python 2-d array form, e.g., [[\"adj1\", \"adj2\", ..., \"adj5\"], [\"adj1\", \"adj2\", ..., \"adj5\"], [\"adj1\", \"adj2\", ..., \"adj5\"]]."
        return res

    def __get_adj_words(self, lst):
        prompt = self.__generate_adj_words_prompt(lst)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        adj_words_list_str = answer_item["text"].strip()
        # print("adj_words_list_str:", adj_words_list_str)
        adj_words_list = eval(adj_words_list_str)
        return adj_words_list
    
    # Get antonym
    def __get_antonym_of_single_adj_word(self, word_item):
        res = None
        if (len(wordnet._morphy(word_item, pos="a")) == 0):
            return res
        
        lemma = wordnet.lemma(word_item + ".a.01." + word_item)
        # print("lemma:", lemma)
        if (lemma is None):
            return res
        synset = lemma.synset()
        if (synset.pos() == "a"):
            antonyms = lemma.antonyms()
            # print("antonyms:", antonyms)
            if (len(antonyms) > 0):
                antom = antonyms[0]
                # print("antom:", antom.name())
                res = antom.name()
        else:
            main_synset = None
            for similar_synset in synset.similar_tos():
                if similar_synset.pos() == 'a':
                    main_synset = similar_synset
                    break

            antonyms = []
            if main_synset:
                for lemma in main_synset.lemmas():
                    for antonym in lemma.antonyms():
                        antonyms.append(antonym.name())

            # print("antonyms:", antonyms)
            if (len(antonyms) > 0):
                res = antonyms[0]
        
        return res
    
    def __get_antonym_by_openai(self, word_item):
        # prompt = f"Find an antonym for the adjective word \"{word_item}\""
        prompt = f"Find an antonym for the adjective word \"{word_item}\". Give result without \".\""
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        antonym_word = answer_item["text"].strip().lower()
        if (antonym_word.endswith(".")):
            antonym_word = antonym_word[:-1]
        return antonym_word
    
    def __get_antonym_word_list(self, lst):
        res = []
        for sub_lst in lst:
            sub_res = []
            for adj_word in sub_lst:
                antonym_word = self.__get_antonym_of_single_adj_word(adj_word)
                if (antonym_word is None):
                    antonym_word = self.__get_antonym_by_openai(adj_word)
                sub_res.append(antonym_word)
            res.append(sub_res)
        return res
    
    # Get anologies from the combination of negative adjective word & the item
    def __combined_antonyms_with_items(self, antonyms, items):
        res = []
        for i in range(len(items)):
            sub_res = []
            sub_antonyms = antonyms[i]
            item = items[i]
            for antonym in sub_antonyms:
                term = (antonym + " " + item)
                sub_res.append(term)
            res.append(sub_res)
        return res
    
    def __get_analogies_by_openai(self, descriptive_terms):
        prompt = f"Like that \"A snail in a swimming pool\" is an analogy to \"slow boat\", what are the things that can be used as anologies to {descriptive_terms}?  Give result in python 2-d array form, e.g., [[\"ans1\", \"ans2\", \"ans3\", \"ans4\", \"ans5\"], [\"ans1\", \"ans2\", \"ans3\", \"ans4\", \"ans5\"], ...]."

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("\nresponse:", response)
        choices = response["choices"]
        answer_item = choices[0]
        anology_list_str = answer_item["text"].strip()
        # print("\nanology_list_str:", anology_list_str)
        anology_list = eval(anology_list_str)
        # print("\nanology_list:", anology_list)
        return anology_list
    
    # Paraphrasing
    # seed_sentence_list
    def __create_seed_sentence_list(self, anology_list, adj_words_list, items):
        res = []
        for i in range(len(items)):
            sub_res = []
            item = items[i]
            sub_adj_words = adj_words_list[i]
            sub_analogies = anology_list[i]        
            for j in range(len(sub_analogies)):
                analogy = sub_analogies[j]
                adj_word = sub_adj_words[j]            
                seed_sentence = f"such a {adj_word} {item}, like {analogy.lower()}"
                sub_res.append(seed_sentence)
            res.append(sub_res)
        return res
    
    def __paraphrase_seed_sentence(self, seed_sentence):
        prompt = f"paraphrase the sentence: \"{seed_sentence}\", in another form, without transition words like \"but\", \"yet\", etc.",

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print("response:", response)
        choices = response["choices"]
        answer_item = choices[0]
        answer_str = answer_item["text"].strip()
        
        return answer_str
    
    def __paraphrase_sentences(self, seed_sentence_list):
        res = []
        for i in range(len(seed_sentence_list)):
            sub_res = []
            sub_seed_sentence_list = seed_sentence_list[i]
            for seed_sentence in sub_seed_sentence_list:
                paraphrased_sentence = self.__paraphrase_seed_sentence(seed_sentence)
                sub_res.append(paraphrased_sentence)
            res.append(sub_res)
        return res
    

if __name__ == '__main__':
    tool_for_elements = Sacarstic_Comments_Gerneration()
    res = tool_for_elements.analyse_image_url("http://n.sinaimg.cn/sinacn15/250/w640h410/20180318/6d63-fyshfur2581706.jpg")
    print("res:", res)