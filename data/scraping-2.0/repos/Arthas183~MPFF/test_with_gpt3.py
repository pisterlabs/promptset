# ----------- 第一步 -----------

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

file_paths = [
    'test.src', 
    'test.tgt', 
    'PTG_generation_task2.txt', 
    'PTG_STYLE_generation_task2.txt'
    ]

text_lists = []

for file_path in file_paths:
    text = read_file(file_path)
    text_lists.append(text)

# 确保每个文件的行数相同
num_lines = len(text_lists[0])
for i in range(1, len(file_paths)):
    if len(text_lists[i]) != num_lines:
        print("Error: The number of lines in the files is not the same.")
        exit(1)


# ----------- 第二步 -----------

import os
import openai
import re
import time
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = ''
 
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# 语义一致性/保留度、风格变换程度、语句流畅度

data = {
    'semantic_ref': 0,
    'semantic_msff': 0,
    'semantic_both': 0,
    'semantic_none': 0,

    'degree_ref': 0,
    'degree_msff': 0,
    'degree_both': 0,
    'degree_none': 0,

    'fluent_ref': 0,
    'fluent_msff': 0,
    'fluent_both': 0,
    'fluent_none': 0,

    'index': 0
    }

def save_data_to_file(filename, data):
    with open(filename, 'w') as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")

def load_data_from_file(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split(': ')
            data[key] = int(value)
    print(data)
    return data

# check标准：语义更接近原始句子，风格变化方式更接近目标句子，流畅度给定几个量化标准，采取两者相互比较的方式
def check(number, test_src, test_tgt, ref_generation, msff_generation):
    global data

    # 语义
    time.sleep(3)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # messages=[
        #     {"role": "system", "content": "You are an excellent English writer. \n\
        #     Please compare the semantic relevance of the following two texts (REF and MSFF) with the original text test.src, based on lexical overlap, topic consistency, sentence structure, etc.:\n\
        #     Please reply with a number between 0 and 100, indicating the percentage score of the semantic relevance of the content,\
        #     (0 indicates that the original text has no semantic connection with the generated text,\
        #     100 indicates that the semantics of the original text and the generated text are exactly the same)\n\
        #     Please provide your response in the following format: \n\
        #     Semantic relevance of REF_Generation to original text: 30,\
        #     Semantic relevance of MSFF_Generation to original text: 20.\n\
        #     Based on these observations, I think a better option is: REF_generation.<|endoftext|>\n\
        #     or:\n\
        #     Semantic relevance of REF_Generation to original text: 40,\
        #     Semantic relevance of MSFF_Generation to original text: 50.\n\
        #     Based on these observations, I think a better option is: MSFF_generation.<|endoftext|>\n\
        #     or:\n\
        #     Semantic relevance of REF_Generation to original text: 90,\
        #     Semantic relevance of MSFF_Generation to original text: 90.\n\
        #     Based on these observations, I think a better option is: REF_generation and MSFF_generation.<|endoftext|>\n\
        #     or:\n\
        #     Semantic relevance of REF_Generation to original text: 10,\
        #     Semantic relevance of MSFF_Generation to original text: 10.\n\
        #     Based on these observations, I think a better option is: None of them.<|endoftext|>\n"},
        #     {"role": "user", "content": f"original_text: {test_src}"},
        #     {"role": "user", "content": f"REF_generation: {ref_generation}"},
        #     {"role": "user", "content": f"MSFF_generation: {msff_generation}"}
        # ],
        messages = [
            {"role": "system", 
             "content": """
             You are a skilled English writer. Your task is to evaluate which of the two texts, REF_Generation or MSFF_Generation, is semantically closer to the original_text. Make your judgment ONLY based on criteria such as lexical overlap, topic consistency, and conveyance of similar information.\n
             DO NOT consider any stylistic or tonal differences, as the generated texts may inherently deviate in style from the original.\n
             Please analyze the texts closely to identify subtle differences. Clearly specify how REF_Generation and MSFF_Generation differ in terms of content, not style. 
             Based on these differences, indicate which one you believe aligns more closely with the original text. Your response should adhere to the following format:\n
             'My reasons are as follows:
             1. The primary difference between REF_Generation and MSFF_Generation is: XXXXXXXXXX.\n
             2. in terms of lexical overlap, XXXXXXXXXX\n
             3. in terms of topic consistency, XXXXXXXXX\n
             4. in terms of conveyance of similar information, XXXXXXXXX\n\
             Please ENSURE that the final conclusion is consistent with the provided reasons.\n\n
             Considering these observations, I believe the text that exhibits greater semantic similarity to the original is: [ REF_Generation | MSFF_Generation | Both MSFF_Generation and REF_Generation | None of them ]<|endoftext|>.'\n"""
             },
            {"role": "system","content": """
             When comparing the semantic similarity between the original_text and the two generations (REF_Generation and MSFF_Generation), focus on the following:\n
             1. Understand the sentiment of key terms in the original_text, such as 'asshole', which is a strong negative term.\n
             2. Directly compare the core sentiments and meanings expressed in both generations to the original text.\n
             3. Identify and match the central theme or message of each text to the original. Remember, it's about how closely they capture the essence of the original_text, not just word overlap.\n
             Now, given the original_text and two generations, determine which generation is more semantically similar to the original_text."
             """},
            {"role": "user", "content": f"original_text: {test_src}"},
            {"role": "user", "content": f"REF_generation: {ref_generation}"},
            {"role": "user", "content": f"MSFF_generation: {msff_generation}"}
        ],
        temperature=0.0
    )

            # """You are a skilled English writer and evaluator. Your primary task is to evaluate the semantic similarity between the original_text and the two generated texts (REF_Generation and MSFF_Generation). Focus on:
            # 1. Lexical overlap: Compare word choices and phrases in each generation with the original_text.
            # 2. Topic consistency: Ensure each generation stays on topic with the original_text.
            # 3. Conveyance of similar information: Look for key sentiments and meanings, especially strong ones like 'not living in' or 'asshole'.
            # 4. Primary differences: Identify the main distinctions between REF_Generation and MSFF_Generation.
            # Note: Ignore stylistic or tonal differences. Concentrate solely on content and its closeness to the original_text.
            # After analyzing, present your findings in this format:
            # 'My reasons are as follows:
            # 1. The primary difference between REF_Generation and MSFF_Generation is: XXXXXXXXXX.
            # 2. In terms of lexical overlap, XXXXXXXXXX.
            # 3. In terms of topic consistency, XXXXXXXXX.
            # 4. In terms of conveyance of similar information, XXXXXXXXX.
            # Considering these observations, I believe the text that exhibits greater semantic similarity to the original is: [ REF_Generation | MSFF_Generation | Both MSFF_Generation and REF_Generation | None of them ].'
            # """
    response_content = response.choices[0].message.content
    print(response)
    last_sentence = response_content.strip().split('\n')[-1]
    print(last_sentence)

    if 'ref_generation' in last_sentence.lower() and 'msff_generation' in last_sentence.lower():
        # semantic_both += 1
        data['semantic_both'] += 1
    elif 'ref_generation' in last_sentence.lower():
        # semantic_ref += 1
        data['semantic_ref'] += 1
    elif 'msff_generation' in last_sentence.lower():
        # semantic_msff += 1
        data['semantic_msff'] += 1
    else:
        data['semantic_none'] += 1
    if data['semantic_both'] + data['semantic_ref'] + data['semantic_msff'] + data['semantic_none'] != number:
        print("semantic mismatch!! check data, and then press any key to continue:")
        a = input()
        data = load_data_from_file("datasave.txt")
    else:
        print(data)
        save_data_to_file("datasave.txt", data)
    with open("results.txt", "a", encoding='utf-8') as f:
        f.write("--------semantic--------\n")
        f.write(f"GPT response: {response} \n")
        f.write(f"Semantic_response_content: \n{str(response_content)} \n")
        f.write(f"last_sentence: {last_sentence} \n")

    # 风格变换程度
    time.sleep(3)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an excellent English writer. Please compare the following two texts (REF_Generation and MSFF_Generation) and determine which one is more style-transferred:\n\
            I will provide you with two texts: source_text and target_text. \n\
            You need to learn the style transfer method from source_text to target_text, and judge which of the following two texts (REF_Generation and MSFF_Generation) is more in line with this style change method from source_text to target_text. \
            Or which of the two texts (REF_Generation and MSFF_Generation) is closer to target_text. \n\
            Please provide your response in the following format: \n\
            'My reasons are as follows:\n\
            XXXXXXXXXX\n\n\
            Based on these observations, I think a better choice is: MSFF_Generation.<|endoftext|>' \n\
            or:\n\
            'My reasons are as follows:\n\
            XXXXXXXXXX\n\n\
            Based on these observations, I think a better choice is: REF_Generation.<|endoftext|>'\n\
            or:\n\
            'My reasons are as follows:\n\
            XXXXXXXXXX\n\n\
            Based on these observations, I think a better choice is: REF_Generation and MSFF_Generation.<|endoftext|>'\n\
            or:\n\
            'My reasons are as follows:\n\
            XXXXXXXXXX\n\n\
            Based on these observations, I think a better choice is: None of them.<|endoftext|>'\n"},
            {"role": "user", "content": f"source_text: {test_src}"},
            {"role": "user", "content": f"target_text: {test_tgt}"},
            {"role": "user", "content": f"REF_generation: {REF_generation}"},
            {"role": "user", "content": f"MSFF_generation: {MSFF_generation}"}
        ],
        temperature=0.0,
        # max_tokens=1
    )
    response_content = response.choices[0].message.content
    print(response)
    last_sentence = response_content.strip().split('\n')[-1]
    print(last_sentence)

    if 'ref_generation' in last_sentence.lower() and 'msff_generation' in last_sentence.lower():
        # degree_both += 1
        data['degree_both'] += 1
    elif 'ref_generation' in last_sentence.lower():
        # degree_ref += 1
        data['degree_ref'] += 1
    elif 'msff_generation' in last_sentence.lower():
        # degree_msff += 1
        data['degree_msff'] += 1
    else:
        data['degree_none'] += 1
    if data['degree_both'] + data['degree_ref'] + data['degree_msff'] + data['degree_none'] != number:
        print("degree mismatch!! check data, and then press any key to continue:")
        a = input()
        data = load_data_from_file("datasave.txt")
    else:
        print(data)
        save_data_to_file("datasave.txt", data)
    with open("results.txt", "a", encoding='utf-8') as f:
        f.write("--------style transfer--------\n")
        f.write(f"GPT response: {response} \n")
        f.write(f"Style_transfer_degree_response_content: \n{response_content} \n")
        f.write(f"last_sentence: {last_sentence} \n")

    # # 风格变换程度
    # time.sleep(3)
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are an excellent English writer. I will provide you with two texts: source_text and target_text. \n"},
    #         {"role": "user", "content": f"source_text: {test_src}"},
    #         {"role": "user", "content": f"target_text: {test_tgt}"},
    #         {"role": "system", "content": "Please compare the following two texts (REF_Generation and MSFF_Generation) and determine which one is more style-transferred:\n"},
    #         {"role": "user", "content": f"REF_generation: {REF_generation}"},
    #         {"role": "user", "content": f"MSFF_generation: {MSFF_generation}"},
    #         {"role": "system", "content": "You need to learn the style transfer method from source_text to target_text, and judge which of the following two texts (REF_Generation and MSFF_Generation) is more in line with this style change method from source_text to target_text. \
    #         Or which of the two texts (REF_Generation and MSFF_Generation) is closest to target_text. \n\
    #         Please provide your response in the following format: \n\
    #         'My reasons are as follows:\n\
    #         1. the style transfer method from source_text to target_text is XXXXXXXXXX\n\
    #         2. I think [REF_Generation | MSFF_Generation] is more in line with this style transfer method from source_text to target_text.\n\
    #         Based on these observations, I think a better choice is: MSFF_Generation.<|endoftext|>' \n\
    #         or:\n\
    #         'My reasons are as follows:\n\
    #         1. the style transfer method from source_text to target_text is XXXXXXXXXX\n\
    #         2. I think [REF_Generation | MSFF_Generation] is more in line with this style transfer method from source_text to target_text.\n\
    #         Based on these observations, I think a better choice is: REF_Generation.<|endoftext|>'\n\
    #         or:\n\
    #         'My reasons are as follows:\n\
    #         1. the style transfer method from source_text to target_text is XXXXXXXXXX\n\
    #         2. I think [REF_Generation | MSFF_Generation] is more in line with this style transfer method from source_text to target_text.\n\
    #         Based on these observations, I think a better choice is: REF_Generation and MSFF_Generation.<|endoftext|>'\n\
    #         or:\n\
    #         'My reasons are as follows:\n\
    #         1. the style transfer method from source_text to target_text is XXXXXXXXXX\n\
    #         2. I think [REF_Generation | MSFF_Generation] is more in line with this style transfer method from source_text to target_text.\n\
    #         Based on these observations, I think a better choice is: None of them.<|endoftext|>'\n\n\
    #         Please ENSURE that the final conclusion is consistent with the provided reasons."}
    #     ],
    #     temperature=0.0
    # )
    # response_content = response.choices[0].message.content
    # print(response)
    # last_sentence = response_content.strip().split('\n')[-1]
    # print(last_sentence)

    # if 'ref_generation' in last_sentence.lower() and 'msff_generation' in last_sentence.lower():
    #     # degree_both += 1
    #     data['degree_both'] += 1
    # elif 'ref_generation' in last_sentence.lower():
    #     # degree_ptg += 1
    #     data['degree_ref'] += 1
    # elif 'msff_generation' in last_sentence.lower():
    #     # degree_ptg_style += 1
    #     data['degree_msff'] += 1
    # else:
    #     data['degree_none'] += 1
    # if data['degree_both'] + data['degree_ref'] + data['degree_msff'] + data['degree_none'] != number:
    #     print("degree mismatch!! check data, and then press any key to continue:")
    #     a = input()
    #     data = load_data_from_file("datasave.txt")
    # else:
    #     print(data)
    #     save_data_to_file("datasave.txt", data)
    # with open("results.txt", "a", encoding='utf-8') as f:
    #     f.write("--------style transfer--------\n")
    #     f.write(f"GPT response: {response} \n")
    #     f.write(f"Style_transfer_degree_response_content: \n{response_content} \n")
    #     f.write(f"last_sentence: {last_sentence} \n")


    # # 风格变换程度 (需要研究)
    # # Please compare the following two texts (PTG_Generation and PTG_STYLE_Generation) and determine which one is more style-transferred:\n
    # time.sleep(3)
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are an excellent English writer. \n\
    #         I will provide you with two texts: source_text and target_text. \n\
    #         You need to learn the style transfer method from source_text to target_text, and judge which of the following two texts (PTG_Generation and PTG_STYLE_Generation) is more in line with this style change method from source_text to target_text. \
    #         Or which of the two texts (PTG_Generation and PTG_STYLE_Generation) is closer to target_text. \n\
    #         Please provide your response in the following format: \n\
    #         'My reasons are as follows:\n\
    #         XXXXXXXXXX\n\n\
    #         Based on these observations, I think a better choice is: PTG_STYLE_Generation.<|endoftext|>' \n\
    #         or:\n\
    #         'My reasons are as follows:\n\
    #         XXXXXXXXXX\n\n\
    #         Based on these observations, I think a better choice is: PTG_Generation.<|endoftext|>'\n\
    #         or:\n\
    #         'My reasons are as follows:\n\
    #         XXXXXXXXXX\n\n\
    #         Based on these observations, I think a better choice is: PTG_Generation and PTG_STYLE_Generation.<|endoftext|>'\n\
    #         or:\n\
    #         'My reasons are as follows:\n\
    #         XXXXXXXXXX\n\n\
    #         Based on these observations, I think a better choice is: None of them.<|endoftext|>'\n"},
    #         {"role": "user", "content": f"source_text: {test_src}"},
    #         {"role": "user", "content": f"target_text: {test_tgt}"},
    #         {"role": "user", "content": f"PTG_generation: {REF_generation}"},
    #         {"role": "user", "content": f"PTG_STYLE_generation: {MSFF_generation}"}
    #     ],
    #     temperature=0.0
    # )
    # response_content = response.choices[0].message.content
    # print(response)
    # last_sentence = response_content.strip().split('\n')[-1]
    # print(last_sentence)

    # if 'ptg_generation' in last_sentence.lower() and 'ptg_style_generation' in last_sentence.lower():
    #     # degree_both += 1
    #     data['degree_both'] += 1
    # elif 'ptg_generation' in last_sentence.lower():
    #     # degree_ptg += 1
    #     data['degree_ref'] += 1
    # elif 'ptg_style_generation' in last_sentence.lower():
    #     # degree_ptg_style += 1
    #     data['degree_msff'] += 1
    # else:
    #     data['degree_none'] += 1
    # if data['degree_both'] + data['degree_ref'] + data['degree_msff'] + data['degree_none'] != number:
    #     print("degree mismatch!! check data, and then press any key to continue:")
    #     a = input()
    #     data = load_data_from_file("datasave.txt")
    # else:
    #     print(data)
    #     save_data_to_file("datasave.txt", data)
    # with open("results.txt", "a", encoding='utf-8') as f:
    #     f.write("--------style transfer--------\n")
    #     f.write(f"GPT response: {response} \n")
    #     f.write(f"Style_transfer_degree_response_content: \n{response_content} \n")
    #     f.write(f"last_sentence: {last_sentence} \n")


    # 语句流畅度
    time.sleep(3)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are an excellent English writer. Please compare the fluency and coherence of the following two texts and determine which one is more fluent and well-structured:\n"},
            {"role": "user", "content": f"REF_generation: {REF_generation}"},
            {"role": "user", "content": f"MSFF_generation: {MSFF_generation}"},
            {"role": "system", "content": f"Please provide your response by indicating which text you find more fluent and well-structured: REF_generation or MSFF_generation.\n\
            Fluency is judged by multiple aspects: \n\
            1. Sentence Structure. \n\
            2. Grammatical correctness: Check whether the grammar of the sentence is correct, including vocabulary usage, subject-verb agreement, tense, etc. \n\
            3. Clarity of Expression: Evaluate whether the sentence conveys the intended message clearly and is easy to understand. \n\
            4. Word Choice: Consider the accuracy and appropriateness of words. \n\
            Additionally, you can provide comments for explaination if needed. \n\
            Please provide your response in the following format:\n\
            'My reasons are as follows:\n\
            1. Sentence Structure: [Your observation and comparison]\n\
            2. Grammatical Correctness: [Your observation and comparison]\n\
            3. Clarity of Expression: [Your observation and comparison]\n\
            4. Word Choice: [Your observation and comparison]\n\n\
            Based on these observations, I think a better choice is: REF_generation.<|endoftext|>'\n\n\
            or:\n\
            'My reasons are as follows:\n\
            1. Sentence Structure: [Your observation and comparison]\n\
            2. Grammatical Correctness: [Your observation and comparison]\n\
            3. Clarity of Expression: [Your observation and comparison]\n\
            4. Word Choice: [Your observation and comparison]\n\n\
            Based on these observations, I think a better choice is: MSFF_generation.<|endoftext|>'\n\n\
            or:\n\
            'My reasons are as follows:\n\
            1. Sentence Structure: [Your observation and comparison]\n\
            2. Grammatical Correctness: [Your observation and comparison]\n\
            3. Clarity of Expression: [Your observation and comparison]\n\
            4. Word Choice: [Your observation and comparison]\n\n\
            Based on these observations, I think a better choice is: Both REF_generation and MSFF_generation.<|endoftext|>'\n\
            or:\n\
            'My reasons are as follows:\n\
            1. Sentence Structure: [Your observation and comparison]\n\
            2. Grammatical Correctness: [Your observation and comparison]\n\
            3. Clarity of Expression: [Your observation and comparison]\n\
            4. Word Choice: [Your observation and comparison]\n\n\
            Based on these observations, I think a better choice is: None of them.<|endoftext|>'\n"
            }
        ],
        temperature=0.0,
        # max_tokens=1
    )
    response_content = response.choices[0].message.content
    print(response)
    last_sentence = response_content.strip().split('\n')[-1]
    print(last_sentence)

    if 'ref_generation' in last_sentence.lower() and 'msff_generation' in last_sentence.lower():
        # fluent_both += 1
        data['fluent_both'] += 1
    elif 'ref_generation' in last_sentence.lower():
        # fluent_ref += 1
        data['fluent_ref'] += 1
    elif 'msff_generation' in last_sentence.lower():
        # fluent_msff += 1
        data['fluent_msff'] += 1
    else:
        data['fluent_none'] += 1
    if data['fluent_both'] + data['fluent_ref'] + data['fluent_msff'] + data['fluent_none'] != number:
        print("fluent mismatch!! check data, and then press any key to continue:")
        a = input()
        data = load_data_from_file("datasave.txt")
    else:
        print(data)
        save_data_to_file("datasave.txt", data)
    data['index'] += 1
    save_data_to_file("datasave.txt", data)
    with open("results.txt", "a", encoding='utf-8') as f:
        f.write("--------fluency and coherence--------\n")
        f.write(f"GPT response: {response} \n")
        f.write(f"fluency_response_content: \n{response_content} \n")
        f.write(f"last_sentence: {last_sentence} \n")

    
    print("semantic: ", "ref: ", data['semantic_ref'], "msff: ", data['semantic_msff'], "both: ", data['semantic_both'], "none: ", data['semantic_none'])
    print("style transfer degree: ", "ref: ", data['degree_ref'], "msff: ", data['degree_msff'], "both: ", data['degree_both'], "none: ", data['degree_none'])
    print("fluency degree: ", "ref: ", data['fluent_ref'], "msff: ", data['fluent_msff'], "both: ", data['fluent_both'], "none: ", data['fluent_none'])
    with open("results.txt", "a", encoding='utf-8') as f:
        f.write(f"semantic: ref: {data['semantic_ref']}, msff: {data['semantic_msff']}, both: {data['semantic_both']}, none: {data['semantic_none']}\n")
        f.write(f"style_transfer_degree: ref: {data['degree_ref']}, msff: {data['degree_msff']}, both: {data['degree_both']}, none: {data['degree_none']}\n")
        f.write(f"fluency_degree: ref: {data['fluent_ref']}, msff: {data['fluent_msff']}, both: {data['fluent_both']}, none: {data['fluent_none']}\n")



if __name__ == "__main__":
    if not os.path.exists("datasave.txt"):# 不存在则新建一个，否则读取原来的
        save_data_to_file("datasave.txt", data)
    data = load_data_from_file("datasave.txt")
    # 逐个抽取相同序号的句子并进行实验
    start = data['index']
    for line_index in range(start, num_lines):
        # 从每个列表中抽取相同序号的句子
        sentences = [text_list[line_index] for text_list in text_lists]
        # 打印抽取到的句子
        print("Sentences for line", line_index + 1)
        test_src = sentences[0].strip()
        print("test_src: ", test_src)
        test_tgt = sentences[1].strip()
        print("test_tgt: ", test_tgt)
        REF_generation = sentences[2].strip()
        print("REF_generation: ", REF_generation)
        MSFF_generation = sentences[3].strip()
        print("MSFF_generation: ", MSFF_generation)

        with open("results.txt", "a", encoding='utf-8') as f:
            f.write("-------------------------------------\n")
            f.write(f"Sentences for line {line_index + 1} \n")
            f.write(f"test_src: {test_src} \n")
            f.write(f"test_tgt: {test_tgt} \n")
            f.write(f"REF_generation: {REF_generation} \n")
            f.write(f"MSFF_generation: {MSFF_generation} \n")

        print("----------")
        # 在这里可以进行实验，比如进行句子的比较等操作
        check(line_index + 1, test_src, test_tgt, REF_generation, MSFF_generation)
        print("press any key to continue...")
        # a=input()
    print()
