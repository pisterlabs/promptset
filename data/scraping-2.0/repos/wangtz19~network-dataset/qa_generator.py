import os
import openai
import logging
import pandas as pd
import opencc
import time
import re
import argparse
from utils import set_proxy, test_proxy, set_openai_key
from tqdm import tqdm
from text2vec import Similarity
from rouge_chinese import Rouge
import jieba


tqdm.pandas()


QUESTION_PROMPT = "请根据以下文本生成问题，尽可能使用简体中文，问题表述需要清晰详细\n\n文本: {context}\n\n问题:\n1."
ANSWER_PROMPT = "根据以下文本生成问题的答案，尽可能使用简体中文\n\n文本: {context}\n\n问题:\n{questions}\n\n答案:\n1."


def gen_questions_by_davinci(row, max_tokens=500):
    try:
        response = openai.Completion.create(
            # engine="davinci-instruct-beta-v3",
            engine="text-davinci-003",
            prompt=QUESTION_PROMPT.format(context=row.context),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
    except Exception as e:
        logging.error(e)
        return ""


def gen_questions_by_chat(row, max_tokens=500):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": QUESTION_PROMPT.format(context=row.context)
                }
            ],
            temperature=0,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(e)
        return ""
    

def gen_answers_by_davinci(row, max_tokens=500):
    try:
        response = openai.Completion.create(
            # engine="davinci-instruct-beta-v3",
            engine="text-davinci-003",
            prompt=ANSWER_PROMPT.format(context=row.context, questions=row.questions),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
    except Exception as e:
        logging.error(e)
        return ""
    

def gen_answers_by_chat(row, max_tokens=500):
    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(context=row.context, questions=row.questions)
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(e)
        return ""
    

def gen_qa(csv_filename, model="gpt", answer_only=False):
    df = pd.read_csv(csv_filename)
    # check fields
    if "context" not in df.columns:
        assert "summary" in df.columns and "content" in df.columns,\
            "csv file must contain 'context', or both 'summary' and 'content' fields"
        df["context"] = "summary: " +  df.summary + "\n" +  "content: " + df.content
    
    if not answer_only:
        # generate questions based on context
        logging.info("generate questions...")
        start_time = time.time()
        if model == "davinci":
            df["questions"] = df.progress_apply(gen_questions_by_davinci, axis=1)
        else:
            df["questions"] = df.progress_apply(gen_questions_by_chat, axis=1)
        df["questions"] = "1." + df.questions
        print("generate questions time: ", time.time() - start_time)

        # save intermediate result
        df.to_csv(csv_filename.replace(".csv", "-questions.csv"), index=False)

    # generate answers based on context and questions
    logging.info("generate answers...")
    start_time = time.time()
    if model == "davinci":
        df["answers"] = df.progress_apply(gen_answers_by_davinci, axis=1)
    else:
        df["answers"] = df.progress_apply(gen_answers_by_chat, axis=1)
    df["answers"] = "1." + df.answers
    print("generate answers time: ", time.time() - start_time)

    # save raw qa
    save_filename = csv_filename.replace(".csv", "-qa-raw.csv")
    df.to_csv(save_filename, index=False)
    return save_filename


converter = opencc.OpenCC('t2s.json')


def split_qa(csv_filename, aspect_list=[]):
    df = pd.read_csv(csv_filename)
    # split raw qa into qa pairs
    question_list, answer_list = [], []
    asp_score_dict = {f"{asp}_score": [] for asp in aspect_list}
    for idx, row in df.iterrows():
        questions = row.questions.split("\n")
        answers = row.answers.split("\n")
        min_len = min(len(questions), len(answers))
        question_list.extend(list(map(lambda x: re.sub(r"^[0-9]+\.", "", x).strip(" \""), questions[:min_len])))
        answer_list.extend(list(map(lambda x: re.sub(r"^[0-9]+\.", "", x).strip(" \""), answers[:min_len])))
        for asp in aspect_list:
            asp_score_dict[f"{asp}_score"].extend(
                [(row[f"{asp}_q_score"] + row[f"{asp}_a_score"]) / 2] * min_len
            )
    
    qa_dict = {"question": question_list, "answer": answer_list}
    for asp in aspect_list:
        qa_dict[f"{asp}_score"] = asp_score_dict[f"{asp}_score"]

    save_filename = csv_filename.replace("-qa-raw.csv", "-qa.csv")
    pd.DataFrame(qa_dict).to_csv(save_filename, index=False)
    return save_filename


sim_model = Similarity()
rouge = Rouge()
chinese_num = "零一二三四五六七八九十"
filter_pattern_list = [
    r"图[%s\d]+" % chinese_num,
    r"表[%s\d]+" % chinese_num,
    r"第.*[节章]",
    r"本节的内容",
    r"本章的内容",
    r"这篇文档",
    r"本文档",
    r"参考文献",
    r"参考资料",
    r"章节",
    r"这个文档",
    r"本[章节]",
    r"这[篇段]文本",
]
filter_pattern = re.compile("|".join(filter_pattern_list))
sub_rules = {
    "www.aidaan.cn": "",
    "爱答案习题答案课件资源网": "",
    "韶关学院信息工程学院骆耀祖整理": "",
    
}


def filter_qa(csv_filename, min_len = 10, max_len = 300, output_format="csv", 
              sim_upper=0.84, rouge_upper=0.75, rouge_lower=0.4,
              to_prompt=True):
    qa_df = pd.read_csv(csv_filename)
    # filter qa pairs
    print("before filter: ", qa_df.shape)
    # remove too short or too long qa pairs
    qa_df = qa_df[(qa_df.question.str.len() >= min_len) & (qa_df.question.str.len() <= max_len)]
    qa_df = qa_df[(qa_df.answer.str.len() >= min_len) & (qa_df.answer.str.len() <= max_len)]
    print("after length filter: ", qa_df.shape)
    # remove questions not end with question mark
    qa_df = qa_df[qa_df.question.str.endswith("？")]
    print("after question mark filter: ", qa_df.shape)
    # remove answers not end with period
    qa_df = qa_df[qa_df.answer.str.endswith("。")]
    print("after period filter: ", qa_df.shape)

    # remove key words representsing figures or tables
    drop_indice = []
    for idx, row in qa_df.iterrows():
        if filter_pattern.search(row.question) or filter_pattern.search(row.answer):
            drop_indice.append(idx)
        else:
            for k, v in sub_rules.items():
                row.question = row.question.replace(k, v)
                row.answer = row.answer.replace(k, v)
            qa_df.loc[idx] = row
    qa_df = qa_df.drop(drop_indice)
    print("after key word filter: ", qa_df.shape)

    # remove duplicated qa pairs
    qa_df = qa_df.drop_duplicates(subset=["question", "answer"])
    print("after duplicate filter: ", qa_df.shape)
    
    qa_df["similarity"] = qa_df.progress_apply(lambda x: sim_model.get_score(x.question, x.answer), axis=1)
    qa_df["rouge"] = qa_df.progress_apply(lambda x: rouge.get_scores(" ".join(jieba.lcut(x.question)), 
                        " ".join(jieba.lcut(x.answer)))[0]["rouge-l"]["f"], axis=1)
    
    # remove qa pairs with high similarity
    qa_df1 = qa_df[qa_df.similarity < sim_upper]
    qa_df2 = qa_df[(qa_df.similarity >= sim_upper) & (qa_df.similarity < 0.9) & (qa_df.rouge < rouge_lower)]
    qa_df = pd.concat([qa_df2, qa_df1])
    print("after similarity filter: ", qa_df.shape)

    # remove qa pairs with high rouge score
    qa_df = qa_df[qa_df.rouge < rouge_upper]
    print("after rouge filter: ", qa_df.shape)

    # convert traditional chinese to simplified chinese
    qa_df.question = qa_df.question.apply(lambda x: converter.convert(x))
    qa_df.answer = qa_df.answer.apply(lambda x: converter.convert(x))

    if to_prompt:
        # rename columns
        qa_df = qa_df.rename(columns={"question": "prompt", "answer": "completion"})

    # save filtered qa
    if output_format == "csv":
        save_filename = csv_filename.replace(".csv", "-filtered.csv")
        qa_df.to_csv(save_filename, index=False)
    else:
        save_filename = csv_filename.replace(".csv", "-filtered.jsonl")
        qa_df.to_json(save_filename, orient="records", lines=True, force_ascii=False)
    return save_filename


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="input csv filename")
parser.add_argument("--model", type=str, default="gpt", help="model name, `davinci` or `gpt`")
parser.add_argument("--proxy", type=str, default=None, help="proxy address")
parser.add_argument("--key_path", type=str, default=".openai-key2", help="openai key path")
parser.add_argument("--answer_only", action="store_true", help="only generate answers")
parser.add_argument("--output_format", type=str, default="csv", help="output format, `csv` or `jsonl`")


def main():
    args = parser.parse_args()

    if args.proxy is not None:
        set_proxy(proxy=args.proxy)
    else:
        set_proxy()
    assert test_proxy(), "proxy is not working"
    set_openai_key(args.key_path)

    save_filename = gen_qa(args.input, model=args.model, answer_only=args.answer_only)
    save_filename = split_qa(save_filename)
    filter_qa(save_filename, output_format=args.output_format)


if __name__ == "__main__":
    main()