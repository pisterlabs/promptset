# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : test.py
@Project  : SelfPolish
@Time     : 2023/5/30 14:41
@Author   : Zhiheng Xi
"""
from ProblemMethod import *
from openai_utils import *
from utils import *
import os
import argparse
import numpy as np
import traceback
import time
import openai




def main(args):

    keys = [
        "Your Key"
        ]
    # os.environ['http_proxy'] = '127.0.0.1:7890'
    # os.environ['https_proxy'] = '127.0.0.1:7890'
    openai.api_key=keys[1]



    # data
    args.dataset_path = get_dataset_path(args.dataset, args.file_type, ori_path="./datasets")
    questions, answers = data_reader(dataset=args.dataset,dataset_path=args.dataset_path)
    qa_pairs = [(questions[idx], answers[idx]) for idx in range(len(questions))]
    print("loading dataset complete. altogether", len(qa_pairs), "questions")
    print("top 10 quesitons: ", questions[: 10])
    print("top 10 answers: ", answers[: 10])


    if args.num_test == -1:
        qa_pairs_test = qa_pairs
    else:
        np.random.seed(args.seed)
        rand_indices = np.random.choice(len(qa_pairs), args.num_test, replace=False)
        qa_pairs_test = [qa_pairs[i] for i in rand_indices]

        print("qa_pairs_test_len:", len(qa_pairs_test))
        #print("rand_indices", rand_indices)


    # set logs
    set_log(args)

    # answer few shot cot
    if args.method == "few_shot":
        cur_answer_method = AnswerFewShot(
            answer_method=args.method,
            eng=args.eng,
            few_shot_prompt_path="prompt/my_prompts/standard/standard_base_{}.txt".format(args.dataset)
        )
    elif args.method=="few_shot_cot":
        cur_answer_method = AnswerFewShotCoT(
            answer_method=args.method,
            eng=args.eng,
            few_shot_prompt_path="prompt/my_prompts/cot/cot_base_{}.txt".format(args.dataset)
        )
    elif args.method=="least_to_most":
        cur_answer_method=AnswerFewShotLtM(
        answer_method = args.method,
        eng = args.eng,
            problem_reducing_prompt_path="prompt/my_prompts/least_to_most/problem_reducing_{}.txt".format(args.dataset),
            problem_solving_prompt_path="prompt/my_prompts/least_to_most/problem_solving_{}.txt".format(args.dataset)
        )

    # Self Polish
    if args.method2=="SP":
        self_polish_method = ProblemSelfPolish(
            problem_method=args.method2,
            final_choose=args.final_choose,
            max_times=args.max_times,
            answer_method=cur_answer_method
        )


    count = 0
    consistent_rewrite_correct = 0.0
    original_correct = 0.0
    converged_correct = 0.0
    vote_correct = 0.0
    first_correct = 0.0
    last_correct = 0.0
    zero_correct = 0.0
    i = 0
    while i < len(qa_pairs_test):
        question = qa_pairs_test[i][0]
        answer = qa_pairs_test[i][1]
        print()
        print("*************************************************************")
        print("*************************************************************")

        time.sleep(1.5)
        original_correctness=False
        consistent_correctness=False
        try:
            original_correctness,consistent_correctness = self_polish_method.forward(
                # question="James wants to swim across a 20-mile lake. He can swim at a pace of 2 miles per hour. He swims 60% of the distance and then rests on an island for half as long as his swimming time. He then finishes the remaining distance while going half the speed. How long does it take him to swim across the lake?",
                # question="Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at the green house, 2 more to the red house, and half of what was left at the orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?",
                question= question,
                # answer="17",
                # answer="18",
                answer=answer,
                direct_answer_trigger_for_fewshot=args.direct_answer_trigger_for_fewshot,
                dataset=args.dataset,
                prompt_index=args.prompt_index
            )

            if original_correctness:
                original_correct+=1
            if consistent_correctness:
                consistent_rewrite_correct+=1
            print("all/original correct/new correct: {}/{}/{}".format(i+1, original_correct,
                                                                      consistent_rewrite_correct))

            i += 1

        except Exception as e:
            print(repr(e))
            traceback.print_exc()
            if "RateLimitError" in repr(e) or "APIConnectionError" in repr(e) or "AuthenticationError" in repr(e):
                print("Exception of Open AI API Key!")
            else:
                print("all/original correct/new correct: {}/{}/{}".format(i+1, original_correct,
                                                                          consistent_rewrite_correct))

                i += 1
            continue




    pass


def set_log(args):
    if args.method == "least_to_most" and args.method2 == "Normal":
        args.output_csv_path = "./results/{}_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}_prompt_index{}.csv".format(
            args.method2, args.method, args.eng, args.dataset, args.num_test,
            args.seed, args.max_tokens, args.file_type, args.prompt_index
        )

        if not os.path.exists(args.output_csv_path):
            with open(args.output_csv_path, 'a+', encoding='utf-8') as f:
                f.write("")
        make_print_to_file("./logs/{}_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}_prompt_index{}.log".format(
            args.method2, args.method, args.eng, args.dataset, args.num_test,
            args.seed, args.max_tokens, args.file_type, args.prompt_index
        ))

    elif args.method == "least_to_most" and args.method2 == "SP":
        args.output_csv_path = "./results/{}_{}_{}_max_times_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}_prompt_index{}.csv".format(
            args.method2, args.final_choose, args.method, args.max_times, args.eng, args.dataset, args.num_test,
            args.seed, args.max_tokens, args.file_type, args.prompt_index
        )

        if not os.path.exists(args.output_csv_path):
            with open(args.output_csv_path, 'a+', encoding='utf-8') as f:
                f.write("")
        make_print_to_file(
            "./logs/{}_{}_{}_max_times_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}_prompt_index{}.log".format(
                args.method2, args.final_choose, args.method, args.max_times, args.eng, args.dataset, args.num_test,
                args.seed, args.max_tokens, args.file_type, args.prompt_index
            ))

    else:
        if args.method2 == "SP":
            args.output_csv_path = "./results/{}_{}_{}_max_times_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}_prompt_index{}.csv".format(
                args.method2, args.final_choose, args.method, args.max_times, args.eng, args.dataset, args.num_test,
                args.seed, args.max_tokens, args.file_type, args.prompt_index
            )

            if not os.path.exists(args.output_csv_path):
                with open(args.output_csv_path, 'a+', encoding='utf-8') as f:
                    f.write("")

            make_print_to_file(
                "./logs/{}_{}_{}_max_times_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}_prompt_index{}.log".format(
                    args.method2, args.final_choose, args.method, args.max_times, args.eng, args.dataset, args.num_test,
                    args.seed, args.max_tokens, args.file_type, args.prompt_index
                ))
        elif args.method2 == "Normal":
            args.output_csv_path = "./results/{}_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}_prompt_index{}.csv".format(
                args.method2, args.method, args.eng, args.dataset, args.num_test,
                args.seed, args.max_tokens, args.file_type, args.prompt_index
            )

            if not os.path.exists(args.output_csv_path):
                with open(args.output_csv_path, 'a+', encoding='utf-8') as f:
                    f.write("")
            make_print_to_file("./logs/{}_{}_{}_{}_num_test_{}_seed_{}_max_tokens_{}_file_type_{}.log".format(
                args.method2, args.method, args.eng, args.dataset, args.num_test,
                args.seed, args.max_tokens, args.file_type
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", default=None, type=str, help="directory to prompt file (.txt)")
    parser.add_argument("--eng", default="text-davinci-003", type=str,  help="engine")
    parser.add_argument("--dataset", default="gsm8k", type=str,
                        help="the dataset name")
    parser.add_argument("--num_test", default=200, type=int, help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float, help="temperature for generation")
    parser.add_argument("--max_tokens", default=1024, type=int, help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str,
                        help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix", default="", type=str, help="")
    parser.add_argument("--direct_answer_trigger_for_fewshot", default="The answer is", type=str,
                        help="used for extract answer")
    parser.add_argument("--method", default="few_shot", type=str,
                        help="we use prompt so that the method is few-shot")

    parser.add_argument("--method2", default="Normal", type=str,
                        help="SP: Self-Polish；Normal")

    parser.add_argument("--q1", default="ori", type=str,choices=["ori","complex"],
                        help="q1 is used for base prompt. ori for standard prompt and CoT; compelx for Complex CoT")
    parser.add_argument("--sample", default=1, type=int,
                        help="sample path number")
    parser.add_argument("--sleep", default=1, type=int,
                        help="sleep time after error.")
    parser.add_argument("--output_csv_path", default="./results/test_cot_gsm8k_003.csv", type=str)
    parser.add_argument("--file_type", default="test", type=str)
    parser.add_argument("--max_times", default=4, type=int)
    parser.add_argument("--final_choose", default="original", type=str)
    parser.add_argument("--prompt_index", default="0", type=str, help="choose the best prompt set for the dataset")

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--total_ranks", default=1, type=int)

    args = parser.parse_args()
    main(args)
