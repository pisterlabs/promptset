import openai
import os
from dotenv import load_dotenv
from utils import *
import re
import warnings

# word를 만드는데 발생하는 특정 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)


# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")


def get_saved_line_count(output_file_path):
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except FileNotFoundError:
        return 0


def to_paragraphed(original_file_path="test.txt", output_file_path="paragraphed.txt"):
    saved_line_count = get_saved_line_count(output_file_path)

    with open(original_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    joined_text = ''.join([line.strip() for line in lines[saved_line_count:]])
    joined_text = remove_newlines(joined_text) # 이걸 추가함
    paragraphs = split_text_into_paragraphs(joined_text)

    with open(output_file_path, "a", encoding="utf-8") as f:
        for paragraph in paragraphs:
            print(paragraph) #for debugging purposes
            f.write(paragraph + '\n')


def to_corrected(original_file_path="paragraphed.txt", output_file_path="corrected.txt"):
    saved_line_count = get_saved_line_count(output_file_path)

    with open(original_file_path, "r", encoding="utf-8") as f:
        paragraphed_contents = f.read().split('\n')

    for paragraph in paragraphed_contents[saved_line_count:]:
        print('len(paragraph)', len(paragraph))
        if len(paragraph) > 5:
            print('original', paragraph) #for debugging purposes
            corrected_paragraph = correct_text(paragraph)
            print(corrected_paragraph) #for debugging purposes
            corrected_paragraph = corrected_paragraph.replace('\n', '')
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(corrected_paragraph + '\n')


def to_englished(original_file_path="corrected.txt", output_file_path="englished.txt"):
    saved_line_count = get_saved_line_count(output_file_path)

    with open(original_file_path, "r", encoding="utf-8") as f:
        corrected_contents = f.read().split('\n')

    for paragraph in corrected_contents[saved_line_count:]:
        if len(paragraph) > 5:
            print('starting...') #for debugging purposes
            englished_paragraph = englished_text(paragraph)
            print(englished_paragraph) #for debugging purposes
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(englished_paragraph + '\n')


def to_summarised(original_file_path="englished.txt", output_file_path='summarised.txt'):

    with open(original_file_path, "r", encoding="utf-8") as f:
        englished_contents = f.read().split('\n')

    englished_contents = concatenate_three_elements(englished_contents)

    for paragraph in englished_contents:
        print('len(paragraph)', len(paragraph))
        print(paragraph)
        if len(paragraph) > 5:
            summarised_paragraph = summarised_text(paragraph)
            print(summarised_paragraph)

            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(summarised_paragraph + '\n')


if __name__ == '__main__':
    # to_paragraphed()
    # print('to_corrected starting...')
    # to_corrected(original_file_path='intermediate/해부_1_1_1_paragraphed.txt', output_file_path='intermediate/해부_1_1_1_corrected.txt')
   
    # to_englished(original_file_path='intermediate/해부_1_1_1_corrected.txt', output_file_path='intermediate/해부_1_1_1_englished.txt')
    # to_summarised(original_file_path="intermediate/해부_1_1_1_englished.txt", output_file_path='해부_1_1_1_summarised.txt')

    # to_corrected(output_file_path="corrected_line_by_line.txt")
    to_summarised(original_file_path="englished/해부학_2주차_2_다리6_englished.txt", output_file_path='summary/해부학_2주차_2_다리6_summarised.txt')
    print('summarised')
    convert_markdown_to_word('summary/해부학_2주차_2_다리6_summarised.txt', "summary/해부학_2주차_2_다리6_summarised")
