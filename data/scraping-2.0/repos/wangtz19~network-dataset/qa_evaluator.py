import os
import pandas as pd
from gpt_score.gpt3_score import gpt3score
import argparse
from utils import set_proxy, test_proxy, set_openai_key
from qa_generator import split_qa, filter_qa
from tqdm import tqdm
import re
import openai
import logging


tqdm.pandas()

SCORE_TYPES = ["gpt_score", "llm_judge", "llm_judge_ref"]

# QUESTION_PROMPT = "请根据以下文本生成问题,尽可能使用简体中文,{aspect}\n\n文本: {context}\n\n问题:\n1."
# ANSWER_PROMPT = "请根据以下文本生成答案,尽可能使用简体中文,{aspect}\n\n文本: {context}\n\n问题: {question}\n\n答案:\n"

# aspect_question_dict = {
#     "informative": "生成的问题应当覆盖文本的关键信息",
#     "coherent": "生成的问题应当合乎逻辑",
#     "relevant": "生成的问题应当与文本相关",
#     "fluent": "生成的问题应当符合语法且读写流畅"
# }

# aspect_answer_dict = {
#     "informative": "生成的答案应当覆盖文本和问题的关键信息",
#     "coherent": "生成的答案应当合乎逻辑",
#     "relevant": "生成的答案应当与文本和问题相关",
#     "fluent": "生成的答案应当符合语法且读写流畅"
# }

# assert aspect_answer_dict.keys() == aspect_question_dict.keys()
# aspect_list = list(aspect_answer_dict.keys())


GPT_SCORE_ANSWER_PROMPT = """请根据以下文本生成给定问题的答案，你的答案应该考虑与文本和问题的相关性、准确性，同时要考虑答案自身的有用性、深度、创造力和详细程度等因素。
[文本]: 
{context}

[问题]: 
{question}

[答案]:

"""


def eval_answer_by_gpt_score(row):
    input = GPT_SCORE_ANSWER_PROMPT.format(
        context=row["context"],
        question=row["question"]
    )
    output = row["answer"]
    return gpt3score(input, output, gpt3model="davinci003")


LLM_JUDGE_ANSWER_PROMPT = """[指令]
请充当一个公正的裁判，评估AI助手对下面显示的问题的回答质量。你的评估应该考虑回复的有用性、相关性、准确性、深度、创造力和详细程度等因素。请通过提供简短的解释来开始你的评估，并尽可能做到客观。提供解释后，你必须遵循以下格式对回复进行评分(从1到10)：\"[[评分]]\"，例如：\"评分：[[5]]\"。

[问题]
{question}

[AI回答开始]
{answer}
[AI回答结束]
"""

LLM_JUDGE_REF_ANSWER_PROMPT = """[指令]
请充当一个公正的裁判，评估AI助手的回答与参考答案之间的相似程度。你会获得一个参考答案，以及AI回答。参考答案为AI回答的预期值，你需要为AI回答与参考答案之间的相似程度、表达能力与语句含义进行评价。请通过提供简短的解释来开始你的评估，并尽可能做到客观。提供解释后，你必须遵循以下格式对回复进行评分(从1到10)：\"[[评分]]\"，例如：\"评分：[[5]]\"。

[参考答案开始]
{reference}
[参考答案结束]

[AI回答开始]
{answer}
[AI回答结束]
"""


def eval_answer_by_llm_judge(row, max_tokens=1000):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": LLM_JUDGE_ANSWER_PROMPT.format(
                        question=row["question"],
                        answer=row["answer"]
                    )
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(e)
        return ""

def eval_answer_with_ref_by_llm_judge(row, max_tokens=2000):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": LLM_JUDGE_REF_ANSWER_PROMPT.format(
                        question=row["question"],
                        reference=row["reference"],
                        answer=row["answer"]
                    )
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(e)
        return ""


# def evalute_qa(csv_filename):
#     df = pd.read_csv(csv_filename)
#     for field in ["context", "questions", "answers"]:
#         assert field in df.columns, f"Column {field} not found in {csv_filename}"
    
#     for asp in aspect_list:
#         df[f"{asp}_q_score"] = 0
#         df[f"{asp}_a_score"] = 0
#         for idx, row in df.iterrows():
#             input1 = QUESTION_PROMPT.format(aspect=aspect_question_dict[asp], context=row.context)
#             output1 = row.questions
#             score1 = gpt3score(input1, output1, gpt3model="davinci003")
#             df.loc[idx, f"{asp}_q_score"] = score1

#             input2 = ANSWER_PROMPT.format(aspect=aspect_answer_dict[asp], context=row.context, question=row.questions)
#             output2 = row.answers
#             score2 = gpt3score(input2, output2, gpt3model="davinci003")
#             df.loc[idx, f"{asp}_a_score"] = score2
    
#     save_filename = csv_filename.replace("-qa-raw.csv", "-eval-qa-raw.csv")
#     df.to_csv(save_filename, index=False)
#     return save_filename


def evalute_answer(csv_filename, score_type, need_split=True,
                   output_format="csv", filter_before_eval=True):
    assert score_type in SCORE_TYPES, f"score_type should be one of {SCORE_TYPES}, but got {score_type}"
    df_raw = pd.read_csv(csv_filename)
    if score_type == "gpt_score":
        assert "context" in df_raw.columns, f"Column context not found in {csv_filename}"

    if need_split:
        for field in ["questions", "answers"]:
            assert field in df_raw.columns, f"Column {field} not found in {csv_filename}"
        question_list, answer_list, context_list = [], [], []
        for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
            questions = re.split(r"\d+\.", row.questions)
            questions = [q.strip() for q in questions if q.strip()]
            answers = re.split(r"\d+\.", row.answers)
            answers = [a.strip() for a in answers if a.strip()]
            min_len = min(len(questions), len(answers))
            question_list.extend(questions[:min_len])
            answer_list.extend(answers[:min_len])
            if score_type == "gpt_score":
                context_list.extend([row.context] * min_len)
        df_split = pd.DataFrame(
            {"question": question_list, "answer": answer_list, "context": context_list}
        )
    else:
        for field in ["question", "answer"]:
            assert field in df_raw.columns, f"Column {field} not found in {csv_filename}"
        df_split = df_raw

    if filter_before_eval:
        tmp_filename = csv_filename.replace(".csv", "-tmp.csv")
        df_split.to_csv(tmp_filename, index=False)
        load_filename = filter_qa(tmp_filename, to_prompt=False)
        df_split = pd.read_csv(load_filename)
        os.remove(tmp_filename)

    score_field = "gs_score" if score_type == "gpt_score" else "lj_score"
    if score_type == "gpt_score":
        df_split[score_field] = df_split.progress_apply(eval_answer_by_gpt_score, axis=1)
    elif score_type == "llm_judge":
        df_split["reason"] = df_split.progress_apply(eval_answer_by_llm_judge, axis=1)
        score_pattern = re.compile(r"评分：\[\[(\d+)\]\]")
        df_split[score_field] = df_split["reason"].apply(lambda x: int(score_pattern.search(x).group(1)) if score_pattern.search(x) else -1)
    elif score_type == "llm_judge_ref":
        df_split["reason"] = df_split.progress_apply(eval_answer_with_ref_by_llm_judge, axis=1)
        score_pattern = re.compile(r"评分：\[\[(\d+)\]\]")
        df_split[score_field] = df_split["reason"].apply(lambda x: int(score_pattern.search(x).group(1)) if score_pattern.search(x) else -1)
    else:
        # unreachable
        pass

    if output_format == "csv":
        save_filename = csv_filename.replace(".csv", "-eval.csv")
        df_split.to_csv(save_filename, index=False)
    else:
        save_filename = csv_filename.replace(".csv", "-eval.jsonl")
        df_split.to_json(save_filename, orient="records", lines=True,
                         force_ascii=False)
    return save_filename


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="input csv filename")
parser.add_argument("--proxy", type=str, default=None, help="proxy address")
parser.add_argument("--key_path", type=str, default=".openai-key2", help="openai key path")
parser.add_argument("--output_format", type=str, default="csv", help="output format, `csv` or `jsonl`")
parser.add_argument("--score_type", type=str, default="llm_judge", help="score type, `gpt_score` or `llm_judge` or `llm_judge_ref` ")
parser.add_argument("--no_filter", action="store_true", help="whether to filter before eval, default is True")
parser.add_argument("--need_split", action="store_true", help="whether to split question and answer, default is False")


def main():
    args = parser.parse_args()

    if args.proxy is not None:
        set_proxy(proxy=args.proxy)
    else:
        set_proxy()
    assert test_proxy(), "proxy is not working"
    set_openai_key(args.key_path)
    filter_before_eval = not args.no_filter
    evalute_answer(args.input, score_type=args.score_type, 
                   output_format=args.output_format,
                   filter_before_eval=filter_before_eval,
                   need_split=args.need_split)


if __name__ == "__main__":
    main()