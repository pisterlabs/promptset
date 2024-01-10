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
        review_ls = [r for r in review.strip().split("\n") if r != ""]
        if len(review_ls) == 4:
            score1 = float(review_ls[0].replace("System1: ","").replace("/10",""))
            score2 = float(review_ls[2].replace("System2: ","").replace("/10",""))
            return [score1,score2]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]

def parse_explain(review, translator=None):
    try:
        review_ls = [r for r in review.strip().split("\n") if r != ""]
        if len(review_ls) == 4:
            explain1 = review_ls[1]
            explain2 = review_ls[3]
            if translator is not None:
                explain1 = translator.translate(explain1)
                explain2 = translator.translate(explain2)
            return [explain1,explain2]
        else:
            raise Exception("Invalid explain pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the explain pair."
        )
        return ["ERROR", "ERROR"]



def gen_prompt(ques, ans1, ans2):
    SYSTEM_PROMPT = '''The followings are ChatGPT-like systems' outputs based on a single prompt. Please rate an overall score on a ten point scale for each system and give a short explanation to justify your scores. Please try not to give the same scores for different system unless they are indistinguishable.'''
    USER_PROMPT = '''
    Prompt:
    <prompt-input>

    System1:
    <system1-output>

    System2:
    <system2-output>
    '''
    USER_PROMPT = USER_PROMPT.replace("<prompt-input>", ques)
    USER_PROMPT = USER_PROMPT.replace("<system1-output>", ans1)
    USER_PROMPT = USER_PROMPT.replace("<system2-output>", ans2)
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

