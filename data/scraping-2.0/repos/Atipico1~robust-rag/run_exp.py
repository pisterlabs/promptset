import argparse
from email import generator
import os
from re import L
import wandb
import numpy as np
import pandas as pd
import random
import joblib
from evaluation import get_encoder, build_qa_prompt
from dataset import make_perturb_testset
from src.classes.prompt import TotalPromptSet
from utils import *
from metrics import *
from tqdm import tqdm
from perturbations import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

def evaluate_single_model(dataset: List[PromptSet],
                          model:dict,
                          model_name: str,
                          args: dict):
    prompts, answers, ori_answers = [], [], []
    idx = 0
    print("length of dataset -> ", len(dataset))
    instruction = get_instruction(args["instruction_type"])
    client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
    encoder = get_encoder(args["lm"])
    if args["perturb_testset"]:
        before = len(dataset)
        dataset = make_perturb_testset(dataset, args)
        for d in dataset[:5]:
            print("Q:", d.query)
            print("A:", d.answers)
            print("Substitution:", d.substitution)
            print("Context:", "\n".join([wiki.text for wiki in d.supports]))
            print("-"*100)
        assert len(data) == before, "length of dataset should be same"
    for idx, data in enumerate(dataset):
        data.supports = data.supports[:args["num_wiki"]]
        prompt = build_qa_prompt(data, model, model_name, args, instruction, encoder)
        if idx == 0:
            print(prompt)
        has_answer_in_context = any([wiki.has_answer for wiki in data.supports])
        ori_answers.append(data.answers)
        if args["unanswerable"]:
            if not(has_answer_in_context):
                data.answers = ["unanswerable"]
        answer = data.answers
        prompts.append(prompt)
        answers.append(answer)
    if args["skip_model_output"]:
        joblib.dump(prompts, )
        raise ValueError
    if args["lm"].startswith("Llama"):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/"+args["lm"])
        if "13b" in args["lm"] or "70b" in args["lm"]:
            model = AutoModelForCausalLM.from_pretrained("meta-llama/"+args["lm"], device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained("meta-llama/"+args["lm"]).to(args["device"])
        model.eval()
    elif args["lm"].startswith("mistral"):
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").to(args["device"])
        model.eval()
    is_corrects, f1_scores, num_tokens, has_answers, total_preds, total_candidates  = [], [], [], [], [], []
    with torch.no_grad():
        for i in (tq := tqdm(range(0, len(prompts), args["bs"]), desc=f"EM:  0.0%")):
            b_prompt, b_answer = prompts[i:i+args["bs"]], answers[i:i+args["bs"]]
            if args["lm"] in ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-16k"]:
                preds = generate_answer_from_gpt(b_prompt, client, args["max_tokens"])
            elif args["lm"].startswith("mistral"):
                tokenizer.pad_token_id = 2
                preds = []
                if not args["ensemble"]:
                    output = tokenizer(b_prompt, return_tensors="pt", padding="longest").to(args["device"])
                    output = model.generate(**output, max_new_tokens=args["max_tokens"], pad_token_id=tokenizer.eos_token_id)
                    pred = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for p in pred:
                        preds.append(p.split("[/INST]")[1].strip())
                else:
                    for p in b_prompt:
                        output = tokenizer(p, return_tensors="pt", padding="longest").to(args["device"])
                        output = model.generate(**output, max_new_tokens=args["max_tokens"], pad_token_id=tokenizer.eos_token_id)
                        pred = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        candidates = [i.split("[/INST]")[1].strip() for i in pred]
                        total_candidates.append("||".join(candidates))
                        result = aggregate_ensemble(candidates, args)
                        preds.append(result)
            elif args["lm"].startswith("Llama"):
                preds = []
                if "chat" not in args["lm"]:
                    tokenizer.pad_token = tokenizer.eos_token
                output = tokenizer(b_prompt, return_tensors="pt", padding="longest").to(args["device"])
                output = model.generate(**output, max_new_tokens=args["max_tokens"])
                pred = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if "chat" in args["lm"]:
                    for p in pred:
                        preds.append(p.split("[/INST]")[1].strip())
                else:
                    for _pred, _prompt in zip(pred, b_prompt):
                        generation_str = _pred[len(_prompt):]
                        _answer = generation_str.split("\n")[0]
                        print("-"*100)
                        print("Answer ->", _answer)
                        print("Generation ->", generation_str)
                        print("-"*100)
                        if len(_answer) < 3:
                            preds.append(generation_str)
                        else:
                            preds.append(_answer)
            total_preds.extend(preds)
            is_corrects.extend([exact_match_score(pred, answer, normalize_answer) for pred, answer in zip(preds, b_answer)])
            f1_scores.extend([f1_score(pred, answer, normalize_answer) for pred, answer in zip(preds, b_answer)])
            num_tokens.extend([len(encoder.encode(prompt)) if isinstance(prompt, str) else 0 for prompt in b_prompt])
            has_answers.extend([True if answer != ["unanswerable"] else False for answer in b_answer])
            tq.set_description(f"EM : {sum(is_corrects) / len(is_corrects) * 100:4.1f}% || Acc : {sum([int(text_has_answer(ans, pred)) for ans, pred in zip(ori_answers, total_preds)]) / len(total_preds)*100:4.1f}" )
            print(len(total_preds),len(is_corrects), len(f1_scores), len(num_tokens), len(has_answers))
    raw_data = pd.DataFrame(data={"question":[data.query for data in dataset],
                                  "prompt": prompts if isinstance(prompts[0], str) else ["\n\n\n".join(prompt) for prompt in prompts],
                                  "answers": [", ".join(ans) if len(ans) > 1 else ans[0] for ans in answers],
                                  "ori_answers": [", ".join(ans) if len(ans) > 1 else ans[0] for ans in ori_answers],
                                  "prediction": total_preds,
                                  "candidates": total_candidates if total_candidates != [] else [None]*len(total_preds),
                                  "is_exact_match": is_corrects,
                                  "is_accurate": [int(text_has_answer(ans, pred)) for ans, pred in zip(answers, total_preds)],
                                  "ori_is_exact_match": [exact_match_score(pred, answer, normalize_answer) for pred, answer in zip(total_preds, ori_answers)],
                                  "ori_is_accurate": [int(text_has_answer(ans, pred)) for ans, pred in zip(ori_answers, total_preds)],
                                  "num_tokens": num_tokens,
                                  "num_ctxs": [len(data.supports) for data in dataset]
                                  })
    metrics = cal_metrics(raw_data)
    output = pd.DataFrame(data={"em": round(sum(is_corrects) / len(is_corrects) * 100,3),
                                "ori_em": round(sum([exact_match_score(pred, answer, normalize_answer) for pred, answer in zip(total_preds, ori_answers)]) / len(total_preds) * 100, 3),
                                "f1": round(sum(f1_scores) / len(f1_scores) * 100, 3),
                                "accuracy": round(sum([int(text_has_answer(ans, pred)) for ans, pred in zip(answers, total_preds)]) / len(total_preds) * 100, 3),
                                "em_answerable":metrics["em_answerable"],
                                "em_unanswerable":metrics["em_unanswerable"],
                                "acc_answerable":metrics["acc_answerable"],
                                "acc_unanswerable":metrics['acc_unanswerable'],
                                "ori_accuracy": round(sum([int(text_has_answer(ans, pred)) for ans, pred in zip(ori_answers, total_preds)]) / len(total_preds) * 100, 3),
                                "unanswerable_ratio": 100 - round(sum(has_answers) / len(has_answers) * 100, 3),
                                "avg_num_tokens": round(sum(num_tokens) / len(num_tokens), 3),
                                "num_dataset": len(dataset)},
                                index=[0])
    return raw_data, output

def evaluate(dataset: TotalPromptSet, model:dict,  metadata: dict, wandb_run=None):
    raw_data, output = evaluate_single_model(dataset=dataset.prompt_sets,
                                       model=model,
                                       model_name=metadata["model_name"],
                                       args=metadata)
    try:
        raw_data.to_csv("raw_data.csv")
        output.to_csv("output.csv")
    except:
        pass
    tbl_result = wandb.Table(dataframe=output)
    tbl_prompt = wandb.Table(dataframe=raw_data)
    wandb_run.log({"result":tbl_result, "raw-data":tbl_prompt})
    #wandb.log({"result":tbl_result, "raw-data":tbl_prompt})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_wiki",type=int, required=True, help=f"Number of retrieved wiki passages.", dest="num_wiki")
    parser.add_argument("--fewshot_type",type=str, required=False, help="Type of fewshot examples",choices=["cbr", "random"], dest="fewshot_type")
    parser.add_argument("--filter",type=str2bool, required=False, help="1 means fewshots only including entities", dest="filter")
    parser.add_argument("--size",type=int, required=False, default=10, dest="nq_size")
    parser.add_argument("--except_perturb", type=str, required=False, default="", choices=[])
    parser.add_argument(
        "--ex_type", type=str, required=False, help="""["random_top1", "random_exact", "random_unanswerable", "cbr_top1", "cbr_exact", "cbr_unanswerable", "fixed_top1", "fixed_exact", "fixed_unanswerable"]""")
    parser.add_argument("--prompt",type=str, required=False, default="base", dest="instruction_type")
    parser.add_argument("--device",type=str, required=False, default="cuda", dest="device")
    parser.add_argument("--num_examples",type=int, required=False, default=5, dest="num_examples")
    parser.add_argument("--unanswerable",type=str2bool, default=True)
    parser.add_argument("--unanswerable_cnt", type=int, default=1)
    parser.add_argument("--model_name", type=str, choices=["ours","baseline", "hybrid"])
    parser.add_argument(
        "--lm", type=str, default="gpt-3.5-turbo-instruct",
        choices=["gpt-3.5-turbo-instruct", "gpt-3.5-turbo-16k", "mistral-instruct", "Llama-2-7b-chat-hf", "Llama-2-7b-hf", "Llama-2-13b-chat-hf", "Llama-2-13b-hf", "Llama-2-70b-chat-hf", "Llama-2-70b-hf"])
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--perturb_testset", required=False, type=str2bool, default=False)
    parser.add_argument("--perturb_testset_op", required=False, type=str, default="random")
    parser.add_argument("--perturb_data_ratio", required=False, type=float, default=0.5)
    parser.add_argument("--perturb_context_ratio", required=False, type=float, default=0.5)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--skip_model_output", type=str2bool, default=False)
    parser.add_argument("--prompt_test", type=str2bool, default=False)
    parser.add_argument("--skip_wandb", type=str2bool, default=False)
    parser.add_argument("--data_dir", type=str, default="/data/seongil/datasets/")
    parser.add_argument("--data_file", type=str, default="TotalPromptSet-all_ex_20.joblib")
    parser.add_argument("--fixed_set", type=str2bool, default=False)
    parser.add_argument("--adaptive_perturbation", type=str2bool, default=False)
    parser.add_argument("--adaptive_perturbation_type", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--prefix", type=str, default="", required=False)
    parser.add_argument("--ensemble", type=str2bool, default=False)
    parser.add_argument("--ensemble_method", type=str, default="voting", choices=["voting", "avg"])
    parser.add_argument("--formatting", type=str2bool, default=False)
    parser.add_argument("--selective_perturbation", nargs="+", default="", help="swap_entity, adversarial, conflict, swap_context, original")
    args = parser.parse_args()
    metadata = vars(args)
    path = args.data_dir + args.data_file
    total = joblib.load(path)
    total_promptset, total_metadata = total["promptset"], total["metadata"]
    print("length of total promptset -> ", len(total_promptset.prompt_sets))
    metadata["dataset_config"] = total_metadata
    random.seed(42)
    del total
    if not args.skip_wandb:
        run = wandb.init(
        project="rag",
        notes="experiment",
        tags=["baseline"],
        name=make_exp_name(args),
        config=metadata)
    model = dict()
    if args.adaptive_perturbation:
        from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer, pipeline
        qa_model = "deepset/roberta-base-squad2"
        nli_model = "cross-encoder/qnli-electra-base"
        model["nlp"] = pipeline("question-answering", model=qa_model, tokenizer=qa_model, device=args.device)
        model["nli"] = {"model": AutoModelForSequenceClassification.from_pretrained(nli_model).to(args.device),
                        "tokenizer": AutoTokenizer.from_pretrained(nli_model)}
    if not args.test:
        evaluate(total_promptset, model, metadata, run)
    else:
        total_promptset.prompt_sets = random.sample(total_promptset.prompt_sets, args.nq_size)
        evaluate(total_promptset, model, metadata, run)
