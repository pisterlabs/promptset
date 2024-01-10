'''
打分模块
'''
import os
import time
from translate import Translator
import openai
import logging
from llmeval.file_io import load_questions,read_jsonl,write_jsonl,write_excel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_eval(sys_prompt, user_prompt: str, max_tokens: int ,
             model = "gpt-4", max_api_retry=5, temperature=0.2 ):
    logging.basicConfig(level=logging.INFO)
    for i in range(max_api_retry):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f"Failed after {max_api_retry} retries.")
    return "error"



def parse_score(review):
    try:
        score_pair = review.strip().split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]

def parse_explain(review, translator=None):
    if translator is None:
        return review, review
    else:
        sent_ls = review.split(".")
        trans_res = []
        for sent in sent_ls:
            explain_zh = translator.translate(sent)
            trans_res.append(explain_zh)
        review_zh = ".".join(trans_res)
        return review, review_zh

def gen_prompt(ques, ans1, ans2):
    SYSTEM_PROMPT = '''You are a helpful and precise assistant for checking the quality of the answer.'''
    USER_PROMPT_TEMPLATE ='''
    [Question]
    {question}
    
    [The Start of Assistant 1's Answer]
    {answer_1}
    [The End of Assistant 1's Answer]
    
    [The Start of Assistant 2's Answer]
    {answer_2}
    
    [The End of Assistant 2's Answer]
    
    [System]
    We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
    Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
    '''
    USER_PROMPT = USER_PROMPT_TEMPLATE.format(question=ques, answer_1 = ans1, answer_2 = ans2)
    return SYSTEM_PROMPT, USER_PROMPT



def review_inference(args, infer_name):
    # 1 read data
    question_jsons = load_questions(args.input)
    # answer1 = llm out, answer2 = chatgpt out
    answer1_file = os.path.join(args.outdir, f"answer_{infer_name}.jsonl")
    answer2_file = os.path.join(args.outdir, f"answer_gpt35.jsonl")
    answer1_jsons = read_jsonl(answer1_file)
    answer2_jsons = read_jsonl(answer2_file)
    # check if # of questions, answers are the same
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    # 2 call gpt4
    review_jsons = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))
    translator = Translator(to_lang="zh") if args.tozh else None
    for i in question_idx_list:
        assert (
            answer1_jsons[i]["question_id"]
            == question_jsons[i]["question_id"]
            == answer2_jsons[i]["question_id"]
        )

        ques = question_jsons[i]["text"]
        ans1 = answer1_jsons[i]["text"]
        ans2 = answer2_jsons[i]["text"]
        sys_prompt, user_prompt = gen_prompt( ques, ans1, ans2)

        review = get_eval(sys_prompt, user_prompt, max_tokens=args.max_tokens,
                          model=args.review_model, max_api_retry=args.max_api_retry,
                          temperature= args.temperature)
        score1, score2 = parse_score(review)
        explain1, explain2 = parse_explain(review, translator= translator)

        # review_id = shortuuid.uuid()
        # qid, llm_ans, llm_explain, llm_score,
        # gpt35_ans,gpt35_explain ,gpt35_score
        review_jsons.append(
            {
                "question_id": question_jsons[i]["question_id"],
                "question": question_jsons[i]["text"],
                "llm_answer": ans1,
                "llm_explain": explain1,
                "llm_score": score1,
                "gpt35_answer": ans2,
                "gpt35_explain": explain2,
                "gpt35_score": score2,
            }
        )

    # 3 save result
    outfile_js = os.path.join(args.outdir, f"review_{infer_name}.jsonl")
    outfile_excel = os.path.join(args.outdir, f"review_{infer_name}.xlsx")
    write_jsonl(review_jsons, outfile_js)
    write_excel(review_jsons, outfile_excel)

    return review_jsons

