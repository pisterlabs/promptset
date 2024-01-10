from os import getenv
import logging
from time import sleep
from tqdm import tqdm
from .LLM_task import LLM_task
from src.srt_util.srt import split_script


def get_translation(srt, model, video_name, prompt = None, chunk_size = 1000):
    # print(srt.get_source_only())
    script_arr, range_arr = split_script(srt.get_source_only(),chunk_size)
    translate(srt, script_arr, range_arr, model, video_name, task=prompt)
    pass

def check_translation(sentence, translation):
    """
    check merge sentence issue from openai translation
    """
    sentence_count = sentence.count('\n\n') + 1
    translation_count = translation.count('\n\n') + 1

    if sentence_count != translation_count:
        return False
    else:
        return True

# TODO{david}: prompts selector
def prompt_selector(src_lang, tgt_lang, domain):
    language_map = {
        "EN": "English",
        "ZH": "Chinese",
        "ES": "Spanish",
        "FR": "France",
        "DE": "Germany",
        "RU": "Russian",
        "JA": "Japanese",
        "AR": "Arabic",
    }
    try:
        src_lang = language_map[src_lang]
        tgt_lang = language_map[tgt_lang]
    except:
        print("Unsupported language, is your abbreviation correct?")
        logging.info("Unsupported language detected")
    prompt = f"""
        you are a translation assistant, your job is to translate a video in domain of {domain} from {src_lang} to {tgt_lang}, 
        you will be provided with a segement in {src_lang} parsed by line, where your translation text should keep the original 
        meaning and the number of lines.
        """
    return prompt

def translate(srt, script_arr, range_arr, model_name, video_name=None, attempts_count=5, task=None, temp = 0.15):
    """
    Translates the given script array into another language using the chatgpt and writes to the SRT file.

    This function takes a script array, a range array, a model name, a video name, and a video link as input. It iterates
    through sentences and range in the script and range arrays. If the translation check fails for five times, the function
    will attempt to resolve merge sentence issues and split the sentence into smaller tokens for a better translation.

    :param srt: An instance of the Subtitle class representing the SRT file.
    :param script_arr: A list of strings representing the original script sentences to be translated.
    :param range_arr: A list of tuples representing the start and end positions of sentences in the script.
    :param model_name: The name of the translation model to be used.
    :param video_name: The name of the video.
    :param attempts_count: Number of attemps of failures for unmatched sentences.
    :param task: Prompt.
    :param temp: Model temperature.
    """

    if input is None: 
        raise Exception("Warning! No Input have passed to LLM!")
    if task is None:
        task = "你是一个翻译助理，你的任务是翻译视频，你会被提供一个按行分割的英文段落，你需要在保证句意和行数的情况下输出翻译后的文本。"
    logging.info(f"translation prompt: {task}")
    previous_length = 0
    for sentence, range_ in tqdm(zip(script_arr, range_arr)):
        # update the range based on previous length
        range_ = (range_[0] + previous_length, range_[1] + previous_length)
        # using chatgpt model
        print(f"now translating sentences {range_}")
        logging.info(f"now translating sentences {range_}")
        flag = True
        while flag:
            flag = False
            try:
                translate = LLM_task(model_name, sentence, task, temp)
            except Exception as e:
                logging.debug("An error has occurred during translation:", e)
                logging.info("Retrying... the script will continue after 30 seconds.")
                sleep(30)
                flag = True

        logging.info(f"source text: {sentence}")
        logging.info(f"translate text: {translate}")
        srt.set_translation(translate, range_, model_name, video_name)
