import hanzipy as hanzi
# import dictionary
from hanzipy.decomposer import HanziDecomposer
decomposer = HanziDecomposer()
# import decomposer
from hanzipy.dictionary import HanziDictionary
dictionary = HanziDictionary()
from collections import defaultdict
from secret_openai_apikey import api_key
import openai
openai.api_key = api_key
import utils
import re
import ast
import jieba
import logging
from unidecode import unidecode
import copy

def chinese_stopwords():
    return ["、","。","〈","〉","《","》","一","一个", ".", ",", "!", ";", ":"] 

def chatgpt_get_pinyin(word):
    """e.g., returns ['shou4', 'bu4', 'liao3'], given input 受不了了]"""
    prompt = f"""Reply with NOTHING except the pinyin for the following word: {word}, formatted as a python list (e.g.,  ["shou4", "bu4", "liao3", "le5"]). If there are multiple options, take your best guess based on context."""
    #print("user: ", prompt)
    response = utils.get_chatgpt_response([{f"role":"user", "content":prompt}], temperature=0)
    #print("response: ", response)
    return ast.literal_eval(response)

punctuation = r"""!?.;:。。()[]？。、；："`'/<>,=$~‘-！°^*• 『""" 
def has_punctuation(s):  
    return any([c in punctuation for c in s])

def remove_spaces_punctuation(input_string):
    # Remove spaces
    #input_string_no_space_punc = input_string.replace(" ", "")
    input_string_no_space_punc = input_string
    for c in input_string_no_space_punc:
        if c in punctuation:
            input_string_no_space_punc = input_string_no_space_punc.replace(c, "")
    
    # Remove punctuation
    input_string_no_space_punc = ' '.join(input_string_no_space_punc.split())
    return input_string_no_space_punc

def jieba_segmentize(text, remove_stopwords=True):
    c = copy.copy(text)
    res = jieba.lcut_for_search(c)
    if remove_stopwords:
        res = [r for r in res if r not in chinese_stopwords()]
    return res

def chatgpt_word_segmentize(text):
    """e.g., ideally returns [我, 现在, 受不了, 了], given input 我现在受不了了]"""
    prompt = f"""Reply with NOTHING except the segmentation for the following text: "{text}", formatted as a python list (e.g.,  ['我', '今天', '去', '吃饭', '了']). If there are multiple options, take your best guess based on context. If the input is in traditional, your output will be in traditional too."""
    #print("user: ", prompt)
    while True:
        try:
            response = utils.get_chatgpt_response([{f"role":"user", "content":prompt}], temperature=0)
            #print("response: ", response)
            return ast.literal_eval(response)
        except Exception as e:
            print(e)
            print("Trying again, cGPT gave incorrectly formatted output")

def chatgpt_smartish_segmentize(text):
    prompt = f"""As you understand, a text may be segmented into sentences, sentences may be segemented into phrases, phrases into subphrases, into..., into words, and so on. 
    This is a recursive process. Given some (possibly multi-sentence) text, I want you to create a sort of 'power set' of sufficiently 'atomic' (i.e., would yield strong, concise flash cards)
    segments of the text. This is more of an art than a science: you should use your best judgement to determine what is a 'useful' segment. For example:
    The text "我现在受不了了. 你昨天晚上告诉我一言为定，但是你现在还没来. 你最好马上就来." should include ['我', '现在', '受不了', '了', '你', '昨天', '晚上', '告诉', '我', '一言为定', '但是', '你', '现在', '还', '没', '来', '你', '最好', '马上', '就', '来'].
    But, it should also include ['昨天晚上', '还没来', 还没来, 马上就来], as these are also useful segments. So, one good result might be ['我', '现在', '受不了', '了', '你', '昨天', '晚上', '告诉', '我', '一言为定', '但是', '你', '现在', '还', '没', '来', '你', '最好', '马上', '就', '来', '昨天晚上', '还没来', 还没来, 马上就来].
    It should NOT include '你最好', '但是你现在', or '告诉我', as these segments are not useful. 
    It should NOT include segments such as '你昨天晚上告诉我', as this is too long. Disclude punctuation.

    Format your answer as a Python list. The text is: "{text}".
    """
    prompt = f"""I am created atomic Anki cards for learning Chinese out of the following text: "{text}". Please segmentize the chinese translation of the text into 'atomic' segments, which you think would make for good, targeted, flash cards.
    For example, the a result for the text "我现在受不了了. 你昨天晚上告诉我一言为定，但是你现在还没来. 你最好马上就来" might be ['受不了了', '昨天', '晚上', '昨天晚上', '告诉',  '一言为定', '现在', '还', '没', '来', '还没来', '你', '最好', '马上', '就', '来', '马上就来'].
    Notice how 'useless' words for flash cards weren't included (e.g., 我), while some (but not all) small phrases were (e.g., 马上就来) along with their constituent words (e.g., 马上, 就, 来). This is more of an art than a science, and you should use your best judgement.
    Format your answer as a Python list.

    Also provide the Chinese text in your answer. So, return a Python list of lists [<chinese text>, <list_of_segments>].
    """
    while True:
        try: 
            return ast.literal_eval(utils.get_chatgpt_response([{f"role":"user", "content":prompt}], temperature=0))
        except Exception as e:
            print(e)
            print("Trying again, cGPT gave incorrectly formatted output")
    



def word_decomposition_info(word):
    regularity_scale = {0: "No Regularity", 1: "Exact Match (with tone)", 2: "Syllable Match (without tone)", 3: "Alliterates (similar initials)", 4: "Rhymes (similar finals)"}
    output = defaultdict()
    try:
        decomposition = decomposer.decompose_many(word)
    except hanzi.exceptions.NotAHanziCharacter:
        return None
    radicals = defaultdict(list)
    for k in decomposition.keys():
        radicals[k] = decomposition[k]["radical"]
    phonetic_regularities = defaultdict(list)
    for character in word:
        phonetic_regularities[character] = dictionary.determine_phonetic_regularity(character)
    
    phonetic_info = defaultdict(list)
    chatgpt_predicted_pinyin = chatgpt_get_pinyin(word)
    print(chatgpt_predicted_pinyin)
    for character, pinyin in zip(word, chatgpt_predicted_pinyin): 
        try: 
            #print("pinyin:", pinyin, ", character:", character)
            print(character)
            print(type(character))
            print(pinyin)
            print(type(pinyin))
            if phonetic_regularities[character] is None:
                continue # i think there's a bug in hanzipy?
            print(phonetic_regularities[character])
            phonetic_regularities[character][pinyin]
            #import pdb; pdb.set_trace()


            k = pinyin # looks weird, basically the above tells us if pinyin is a valid key and we only proceed here if so
            print("chatgpt worked")
        except KeyError:
            k = next(iter(phonetic_regularities[character])) #'next(iter(...))' just means 'get the key of the first element of the dict', as we're going to assume that's the correct one
            print("chatgpt failed")
        #print("k:", k)
        info = phonetic_regularities[character][k]
        components = info['component']
        regularities = [regularity_scale[r] + " with " + k if r is not None else None for r in info['regularity']]
        phonetic_pinyin = info['phonetic_pinyin']
        phonetic_info[character] = [(component, pinyin, regularity) for component, regularity, pinyin in zip(components, regularities, phonetic_pinyin)]
    for character in word:
        radicals[character] = [(radical, decomposer.get_radical_meaning(radical)) for radical in radicals[character]] # so, we get a dict of (radical, meaning) pairs 
        
    for character in word:
        output[character] = {"radicals": radicals[character], "phonetic_regularities": phonetic_info[character]}
    return output

def text_decomposition_info(text):
    # output is a list of dicts of same format as that of word_decomposition_info(). the difference is that calling this makes the program more careful about pinyin etc. and hence gives better results
    words = chatgpt_word_segmentize(text)
    print(words)
    output = []
    for word in words:
        decomp_dict = word_decomposition_info(word)
        if decomp_dict is None:
            return output
        for key, value in zip(decomp_dict.keys(), decomp_dict.values()):
            output.append({key: value})
    return output

def chatgpt_get_classifiers(text, context_messages):
    mw_finder_prompt = """Your job is now to list the measure word(s) that are associated with the following word(s). Respond with only a Python list, where each entry is a relevant measure word without pinyin."""
    context_messages.extend([{"role":"system", "content":mw_finder_prompt}, {"role":"user", "content":f"The text is {text}. Remember to format your answer as a Python list of chars."}])
    mw_response = utils.get_chatgpt_response_enforce_python_formatting(context_messages, response_on_fail="No measure word", extra_prompt = "Try again— make sure to format your response as a Python list of chars, and only a python list of chars." , start_temperature=0)
    context_messages = utils.update_chat(context_messages, "assistant", mw_response)
    return ast.literal_eval(mw_response), context_messages
    #while True:
    #    try:
    #        #print("user: ", context_messages[-1]["content"])
    #        mw_response = utils.get_chatgpt_response(context_messages, temperature=0)
    #        context_messages = utils.update_chat(context_messages, "assistant", mw_response)
    #        return ast.literal_eval(mw_response), context_messages
    #    except Exception as e:
    #        print("EXCEPTION", e)
    #        print("Trying again, cGPT gave incorrectly formatted output")

def chatgpt_batch_make_trad_from_simplified(simplified_text_list):
    # purposely don't include context_messages here, because this isn't a relevant part of the conversation
    trad_prompt = """Convert each string given into traditional Chinese (no English, simplified Chinese, or pinyin allowed). Provide your output as similarly formatted Python list (no markdown). If an entry is not simplified chinese, just copy it for the corresponding output entry (e.g., if it says 'No Measure Word' then corresponding output is 'No Measure Word')."""
    temp_messages = [{"role":"system", "content":trad_prompt}, {"role":"user", "content":f"The text is {str(simplified_text_list)}."}]
    return utils.get_chatgpt_response_enforce_python_formatting(temp_messages, extra_prompt="Try again, making sure to format your output as if it were a Python list (but no backticks or anything), and say NOTHING else.", response_on_fail=str(["None"]*len(simplified_text_list)), start_temperature=0)

def chatgpt_make_trad_from_simplified(simplified_text):
    # purposely don't include context_messages here, because this isn't a relevant part of the conversation
    trad_prompt = """Your job is now to convert the provided simplified Chinese text into traditional Chinese. Respond with only the converted text (no English, simplified Chinese, or pinyin allowed). Say "None" if insufficient input is provided."""
    temp_messages = [{"role":"system", "content":trad_prompt}, {"role":"user", "content":f"The text is {simplified_text}."}]
    return utils.get_chatgpt_response(temp_messages, temperature=0)

def chatgpt_translate(english_simplified_or_traditional_text, context_messages=[]):
    # purposely don't include context_messages here, because this isn't a relevant part of the conversation
    trans_prompt = """Your job is to translate the provided text (which may be English, traditional Chinese, or simplified Chinese) into simplified Chinese. Respond with only the translated text. Return the text unchanged if it is provided as simplified Chinese."""
    temp_messages = [{"role":"system", "content":trans_prompt}, {"role":"user", "content":f"The text is {english_simplified_or_traditional_text}."}]
    context_messages.extend(temp_messages)
    r = utils.get_chatgpt_response(context_messages, temperature=0)
    return r, remove_spaces_punctuation(remove_non_chinese_from_string(r, keep_punctuation=True))

def chatgpt_translate_to_english(english_simplified_or_traditional_text, context_messages=[]):
    # purposely don't include context_messages here, because this isn't a relevant part of the conversation
    trans_prompt = """Your job is to translate the provided text (which may be English, traditional Chinese, or simplified Chinese) into English. Respond with only the translated text. Return the text unchanged if it is provided as English."""
    temp_messages = [{"role":"system", "content":trans_prompt}, {"role":"user", "content":f"The text is {english_simplified_or_traditional_text}."}]
    context_messages.extend(temp_messages)
    
    try:
        r = utils.get_chatgpt_response(context_messages, temperature=0)
        if "The text is" in r: # literally because i cannot get chatgpt to not say "The text is" at the start of the response
            r = r.split("The text is")[1]
    except Exception as e:
        print(e)
        print("ERROR IN CGPT TRANSLATE TO ENGLISH... MESSAGES ARE", context_messages)
        # print traceback
        import traceback; traceback.print_exc()
        raise e
    return r

def chatgpt_batch_make_pinyin_from_simplified(simplified_text_list):
    # purposely don't include context_messages here, because this isn't a relevant part of the conversation
    trad_prompt = """Convert each string given into pinyin, and provide your output as a corresponding Python list (no markdown). If a given input entry is insuffient, say "No Pinyin" as the corresponding output entry."""
    temp_messages = [{"role":"system", "content":trad_prompt}, {"role":"user", "content":f"The text is {str(simplified_text_list)}."}]
    return utils.get_chatgpt_response_enforce_python_formatting(temp_messages, extra_prompt="Try again, making sure to format your output as if it were a Python list, and say NOTHING else.", response_on_fail=str(["None"]*len(simplified_text_list)), start_temperature=0)

def chatgpt_make_pinyin_from_simplified(simplified_text):
    # purposely don't include context_messages here, because this isn't a relevant part of the conversation
    pinyin_prompt = """Your job is now to convert the provided simplified Chinese text into pinyin. Respond with only the converted text. If input is 'None', reply 'None'."""
    temp_messages = [{"role":"system", "content":pinyin_prompt}, {"role":"user", "content":f"The text is {simplified_text}."}]
    return utils.get_chatgpt_response(temp_messages, temperature=0)

def remove_non_chinese_from_string(text, keep_punctuation=False):
   if not keep_punctuation:
        return  ''.join([o for o in text if re.search(u'[\u4e00-\u9fff]', o)])  # only keep non chinese entries
   else: 
        return ''.join([o for o in text if re.search(u'[\u4e00-\u9fff]', o) or o in punctuation])



def split_text(text):
    segments = re.split(r'(?<=[\u4e00-\u9fa5])(?=[^\u4e00-\u9fa5])|(?<=[^\u4e00-\u9fa5])(?=[\u4e00-\u9fa5])', text)
    language_tags = ['zh-tw' if re.search(r'[\u4e00-\u9fa5]', segment) else 'en' for segment in segments]
    return segments, language_tags

def is_pinyin_tone_marked(s):
    # The following pattern matches Latin letters followed by diacritics
    pinyin_tone_marked_regex = r"\b([a-zA-ZüÜāēīōūǖĀĒĪŌŪǕáéíóúǘÁÉÍÓÚǗǎěǐǒǔǚǍĚǏǑǓǙàèìòùǜÀÈÌÒÙǛ]+\s*)+\b"
    return re.fullmatch(pinyin_tone_marked_regex, s) is not None

def remove_pinyin_tone_marked_ish(text):
    # Split the text into words
    print("remove_pinyin_tone_marked_ish input is ", text)

    words = text.split()

    # Remove words with diacritics, but preserve chinese characters
    words = [word for word in words if word == unidecode(word) or re.search(u'[\u4e00-\u9fff]', word)]

    # Join words back into a string
    text = ' '.join(words)

    return text


    
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    simplified = "我现在受不了了"
    traditional = "愛情"

    #print(text_decomposition_info(simplified))
    #print(word_decomposition_info(traditional))
    #print("word_segmentize: ", chatgpt_word_segmentize("小高昨天夜里肚子疼的很厉害，我跟他一起去医院看的病."))
    #print("smart_segmentize: ", chatgpt_smartish_segmentize("Little Gao had a terrible stomach ache last night, and I was the one to take him to the hospital."))

    semantic_tags_info_prompt = """Now your job is to, given user's text, classify it into any number (including zero) of the following properties. Please format your answer as a Python list of strings.
    Semantic categories:
    - food and drink
    - travel
    - shopping
    - work and employment
    - family and relationships
    - daily routines
    - entertainment
    - health and wellness
    - environment and sustainability
    - technology
    - culture and customs
    - politics and current events
    - education
    - weather and climate
    - sports and fitness
    - art and literature
    - history and traditions
    - holidays and celebrations
    - travel and tourism
    - movies and television
    - music and audio
    - philosophy and religion
    - cars and transportation
    - animals and wildlife
    - business and finance
    - geography and landmarks
    - fashion and style
    - science and technology
    - math, science and innovation
    - language and linguistics
    - learning chinese
    - Social Media and Internet Culture
    - Hobbies and Interests
    - Career and Professional Development
    - Astronomy and Space
    - Home and Lifestyle
    - Celebrity and Pop Culture
    - Cooking and Cuisine
    - Gardening and Plants
    - Chinese Mythology and Folklore
    - Personal Growth and Self-Improvement
    - Human Rights and Social Issues
    - Public Transport and Infrastructure
    - Outdoor Activities and Adventures
    - Photography and Visual Arts
    - Military and Defense
    - Etiquette and Social Norms
    - Pets and Pet Care
    - Volunteering and Community Service
    - Real Estate and Housing
    - Parenting and Childcare
    - Mental Health and Wellness
    - Elderly Care and Retirement
    - Dating and Relationships
    - Marriage and Weddings
"""
word = "人山人海"
#print(utils.get_chatgpt_response([{"role":"system", "content":semantic_tags_info_prompt}, {"role":"user", "content":f"The text is {word}. Remember to format your answer as a Python list of strings"}], temperature=0))
#print(utils.get_chatgpt_response([ {"role":"user", "content":f"is 到底 a grammar point in chinese? Answer ONLY with 'True' or 'False'"}], temperature=0))
#POS_system_prompt = """Your job is now to identify all parts-of-speech throughout the following text. Keep in mind that some words in Chinese can be multiple parts-of-speech at once, though you should not include duplicates in your response. Please format your answer as a Python list of strings, all lowercase and with no duplicate entries."""
#print(utils.get_chatgpt_response([{"role":"system", "content":POS_system_prompt}, {"role":"user", "content":f"The text is {word}. Remember to format your answer as a Python list of strings"}], temperature=0))
#
#phrase_system_prompt = "Your job is now to identify the what type of syntactic phrase the following text is. The options are CP, TP, VP, NP, PP, AdjP, AdvP, XP, X, chengyu. Respond ONLY with your answer. If you think the text is not a phrase, respond with 'not_a_phrase'."
#print(utils.get_chatgpt_response([{"role":"system", "content":phrase_system_prompt}, {"role":"user", "content":f"The text is {word}"}], temperature=0))

text = "这是一个中文句子but this is english再是中文"
#print(remove_non_chinese_from_string(text))

text =  "this is english  english again" 
#print(split_text(text))

text = "máo)"
#print(is_pinyin_tone_marked(text))

text = "This is a test: 中文 nǐ hǎo, wǒ shì māo more english"
#print(remove_pinyin_tone_marked_ish(text)) # eats everything but punctuation

text = "我现在受不了了. 你昨天晚上告诉我一言为定，但是你现在还没来. 你最好马上就来."
#print(jieba_segmentize(text))



data = [('做', [('亻', 'ren2', 'No Regularity with zuo4'), ('故', 'gu4', 'No Regularity with zuo4'), ('亻', 'ren2', 'No Regularity with zuo4'), ('十', 'shi2', 'No Regularity with zuo4'), ('口', 'kou3', 'No Regularity with zuo4'), ('⺙', None, None)]), ('米', [('木', 'Mu4', 'Alliterates (similar initials) with mi3'), ('木', 'mu4', 'Alliterates (similar initials) with mi3'), ('丷', 'ba1', 'No Regularity with mi3'), ('丷', 'xx5', 'No Regularity with mi3'), ('米', 'Mi3', 'Exact Match (with tone) with mi3'), ('米', 'mi3', 'Exact Match (with tone) with mi3')]), ('飯', [('飠', 'shi2', 'No Regularity with fan4'), ('反', 'fan3', 'Syllable Match (without tone) with fan4'), ('人', 'ren2', 'No Regularity with fan4'), ('丨', 'gun3', 'No Regularity with fan4'), ('丨', 'shu4', 'No Regularity with fan4'), ('彐', 'ji4', 'No Regularity with fan4'), ('乚', 'ya4', 'No Regularity with fan4'), ('丶', 'zhu3', 'No Regularity with fan4'), ('丶', 'zhu3', 'No Regularity with fan4'), ('⺁', None, None), ('又', 'you4', 'No Regularity with fan4')])]


