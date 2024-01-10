import argparse
import json
import os
import random
import openai

from tqdm import tqdm
from qasper_evaluator import get_answers_and_evidence
from dotenv import load_dotenv
from pipeline import utils

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_generated_answer(question, context, model, prompt_filename):
    if model != "gpt-3.5-turbo":
        # question only prompt
        if  prompt_filename == 'qasper_question_only_prompt':
            prompt = utils.construct_prompt(
                filename=f"{prompt_filename}.txt",
                prompt_params_dict={
                    "question": question,
                }
            )
            return utils.get_model_response(prompt, model=model)["choices"][0]['text']
        
        # question and context prompt
        prompt = utils.construct_prompt(
            filename=f"{prompt_filename}.txt",
            prompt_params_dict={
                "question": question,
                "context": context,
            }
        )
        if prompt_filename == 'qasper_cot_prompt':
            response = utils.get_model_response(prompt, model=model)["choices"][0]['text']
            explanation = response.split("Answer:")[0]
            answer = response.split("Answer:")[1]
            return {"answer": answer, "explanation": explanation}
        return utils.get_model_response(prompt, model=model)["choices"][0]['text']
    else:
        # gpt-3.5-turbo model
        if prompt_filename == 'qasper_cot_prompt':
            response = utils.get_chat_model_response(context, question, prompt_filename)
            explanation = response.split("Answer:")[0]
            answer = response.split("Answer:")[1]
            return {"answer": answer, "explanation": explanation}
        return utils.get_chat_model_response(context, question, prompt_filename)

def generate_all_answers():
    q_ids = list(test_questions.keys())

    # Load generated answers if file exists
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            gen_answers = json.load(f)
    else:
        gen_answers = {}

    for q_id in tqdm(test_questions.keys()):
        # check if the answer has already been generated
        if q_id in gen_answers.keys():
            print(f"Answer for {q_id} already generated")
            continue

        question = test_questions[q_id]
        if args.retrieval_method == "random":
            random_q_id = random.choice(q_ids)
            context = " | ".join([text for text in test_answers_and_evidence[random_q_id][0]['evidence'] if "FLOAT SELECTED" not in text])
        elif args.retrieval_method == "gold":
            context = " | ".join([text for text in test_answers_and_evidence[q_id][0]['evidence'] if "FLOAT SELECTED" not in text])
        elif args.retrieval_method == "closed-book":
            context = None
            question = question[:-1] + " in the paper \"" + test_titles[q_id] + "\"?"
        else:
            context = " | ".join([text for text in retrieval_psgs[q_id][:args.top_k_passages] if "FLOAT SELECTED" not in text])
            if args.prompt_filename == 'qasper_cot_prompt':
                context = [text for text in retrieval_psgs[q_id][:args.top_k_passages] if "FLOAT SELECTED" not in text]
        gen_answers[q_id] = get_generated_answer(question, context, args.model, args.prompt_filename)

        # save the generated answers
        with open(RESULTS_FILE, 'w') as f:
            json.dump(gen_answers, f, indent=4)

    return gen_answers

def get_predictions(answers_and_evidence, gen_answers, retrieval_psgs):
    predictions = []
    for q_id, references in answers_and_evidence.items():
        if q_id not in gen_answers:
            continue
        predicted_answer = gen_answers[q_id]
        predicted_answer = "Yes" if predicted_answer.startswith("Yes") else "No" if predicted_answer.startswith("No") else predicted_answer
        golden_evidence = [i['evidence'] for i in references]
        golden_answers = [i['answer'] for i in references]
        if args.retrieval_method != "random" and args.retrieval_method != "gold" and args.retrieval_method != "closed-book":
            predicted_evidence = retrieval_psgs[q_id]
        else:
            predicted_evidence = golden_evidence[0]
        predictions.append({'question_id': q_id, 'predicted_answer': predicted_answer, 'golden_answers': golden_answers, 'predicted_evidence': predicted_evidence})
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--retrieval_method",
        type=str,
        choices=[
            'random',
            'closed-book',
            'gold',
            'bm25',
            'monot5-base-msmarco-10k',
            'monot5-base-msmarco-10k_our_chunks',
            'monot5-3b-msmarco-10k'
        ],
        required=True,
        help="Choose retrieval method",
    )

    parser.add_argument(
        "--prompt_filename", 
        type=str, 
        choices=[
            "qasper_zeroshot_prompt",
            "qasper_fewshot_prompt",
            "qasper_cot_prompt",
            "qasper_question_only_prompt"
        ],
        required=True, 
        help="Filename of QA prompt to use"
    )

    parser.add_argument(
        "--model", 
        type=str,
        choices=[
            "text-davinci-003",
            "code-davinci-002",
            "gpt-3.5-turbo"
        ],
        required=True,
        default="text-davinci-003",
        help="Choose between the OpenAI large language models",
    )

    parser.add_argument(
        "--top_k_passages", 
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="Choose the number of passages to retrieve",
    )
    args = parser.parse_args()

    DATA_PATH = 'data/qasper/'
    RESULT_PATH = 'results/'
    RETRIEVAL_PATH = 'retrieval_passages/'

    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, args.retrieval_method), exist_ok=True)
    os.makedirs(os.path.join(RESULT_PATH, args.retrieval_method, args.model), exist_ok=True)
    RESULTS_FILE = os.path.join(RESULT_PATH, args.retrieval_method, args.model, f'{args.prompt_filename}.json')
    PRED_FILE = os.path.join(RESULT_PATH, args.retrieval_method, args.model, f'{args.prompt_filename}.jsonl')

    retrieval_psgs = None
    if args.retrieval_method != 'random' and args.retrieval_method != 'gold' and args.retrieval_method != 'closed-book':
        with open(os.path.join(RETRIEVAL_PATH, f'{args.retrieval_method}_contents.json'), 'r') as f:
            retrieval_psgs = json.load(f)

    # Load test questions, answers and evidence
    with open(os.path.join(DATA_PATH, 'qasper-test-v0.3.json'), 'r') as f:
        test = json.load(f)
    
    test_questions = {}
    test_titles = {}
    for k, v in test.items():
        for qa in v['qas']:
            q_id = qa['question_id']
            test_questions[q_id] = qa['question']
            test_titles[q_id] = v['title']
    
    test_answers_and_evidence = get_answers_and_evidence(test, text_evidence_only=True)

    assert test_questions.keys() == test_answers_and_evidence.keys()

    # if answers have already been generated, evaluate them
    if os.path.exists(PRED_FILE):
        print(os.system(f"python qasper_evaluator.py --predictions {PRED_FILE} --gold {DATA_PATH}/qasper-test-v0.3.json --text_evidence_only"))
    else:
        gen_answers = generate_all_answers()

        if args.prompt_filename == "qasper_cot_prompt":
            answers = {k : v["answer"] for k, v in gen_answers.items()}
            answers = utils.clean_generated_answers(answers)
            explanations = {k : v["explanation"].replace("Explanation: ", "") for k, v in gen_answers.items()}
            relevant_docs = {k : utils.keep_relevant_docs(explanation, retrieval_psgs[k]) for k, explanation in explanations.items()}
            predictions = get_predictions(test_answers_and_evidence, answers, relevant_docs)
        else:
            answers = utils.clean_generated_answers(gen_answers)
            predictions = get_predictions(test_answers_and_evidence, answers, retrieval_psgs)

        # save the predictions in JSON lines format
        with open(PRED_FILE, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')

        # evaluate the predictions
        print(os.system(f"python qasper_evaluator.py --predictions {PRED_FILE} --gold {DATA_PATH}/qasper-test-v0.3.json --text_evidence_only"))