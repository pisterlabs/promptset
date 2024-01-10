from db import db
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import sentence_bleu
import torch
import time
import random
import sys
import numpy as np
import json
import os
import gc
import openai
from datetime import datetime
from chat_gpt3 import chat_gpt3

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/robert'))
from src.robert.robert_lora import robert
from src.robert.robert_lora import build_finetuned_path


db = db()
chatgpt_model = chat_gpt3()

# Add here all models you want. Set the "test" property to false if
# you dont want that model to be tested in the next run.
test_models = [
    {
        'name': 'robert_1k',
        'desc': 'Trained on 1k base chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_5k',
        'desc': 'Trained on 5k base chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_5k_chat_only',
        'desc': 'Trained on 5k chatting chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_21k_chat_only_para',
        'desc': 'Trained on 5k chatting chatgpt ds + 16k chatting paraphrased',
        'test': True
    },
    {
        'name': 'robert_6k_para_chat',
        'desc': 'Trained on 6k base chatgpt ds + 12k paraphrased + 5k chatting ds',
        'test': True
    },
    {
        'name': 'robert_10k_gpt4all',
        'desc': 'Trained on 10k base gpt4all ds',
        'test': True
    },
    {
        'name': 'robert_10k',
        'desc': 'Trained on 10k base chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_45k_chat_para',
        'desc': 'Trained on 12k base chatgpt ds + 12k paraphrased + 5 chatting ds + 16k paraphrased',
        'test': True
    }
]
base_datasets_count = 1000
chat_datasets_count = 1000
done_models = []
include_rouge = False
include_chatgpt = False
generate_rateable_datasets = True
generate_student_instructions = False
generate_student_dialogs = False
student_instruction = """Formulate an instruction or a question towards Rob about the given input"""
student_dialog = """Proactively continue the dialog provided in the input as the student"""


def get_parameters():
    params = []
    f = open("parameters.txt", "r", encoding='utf-8')
    for line in f.readlines():
        params.append(str(line.strip()))
    return params


def test_instruction_following_capabilities(model_name, my_robert):
    '''Test a model for its instruction following capabilities'''
    # First step: calculate a rogue score. Use chatgpt datasets for that.
    print("\n")
    print("----- Testing instruction following capabilities of " + model_name)

    db_name = "chatgpt"
    if("gpt4all" in model_name):
        db_name = "gpt4all"

    base_datasets = db.get_base_datasets("chatgpt", base_datasets_count)
    print("Going through " + str(base_datasets_count) + " datasets.")
    count = 1
    rouge = ROUGEScore()
    # In these tests, we dont want context or anything. Just instruction
    # following capabilities
    for data in base_datasets:
        target = data['output']
        prediction = my_robert.get_response(data['instruction'], use_context=False)
        progress = "Done with " + str(round(100/base_datasets_count*count, 1)) + "%"
        score = rouge(prediction, target)
        bleu_score = float(sentence_bleu([target], prediction))
        db.insert_rogue_score(score, model_name, data['instruction'],
                              target, prediction, data['input'], bleu_score)

        count = count + 1
        #sys.stdout.write('\r')
        sys.stdout.write('Done with ' + str(count) + ' datasets. ' + progress)
        #sys.stdout.flush()


def test_dialog_capabilities(model_name, my_robert):
    print("\n")
    print("----- Testing dialog capabilities of " + model_name)

    chat_datasets = db.get_chatting_datasets_with_input(chat_datasets_count, False)
    print("Going through " + str(chat_datasets_count) + " datasets.")
    count = 1
    rouge = ROUGEScore()
    for data in chat_datasets:
        # For here, we want to work with the input as context.
        target = data['output']
        my_robert.set_context(data['input'].split('\n'))
        prediction = my_robert.get_response(data['instruction'])
        progress = "Done with " + str(round(100/chat_datasets_count*count, 1)) + "%"
        score = rouge(prediction, target)
        bleu_score = float(sentence_bleu([target], prediction))
        db.insert_rogue_score(score, model_name, data['instruction'],
                              target, prediction, data['input'], bleu_score)

        count = count + 1
        # Decomment the two lines below if you dont want a new line in the console.
        # sys.stdout.write('\r')
        sys.stdout.write('Done with ' + str(count) + ' datasets. ' + progress)
        # sys.stdout.flush()


def build_rateable_dataset(model_name, my_robert):
    print("\n")
    print("----- Building rateable datasets for " + model_name)

    # First, the instructions.
    student_instructions = db.get_student_instructions(9999)
    print("Going through " + str(len(student_instructions)) + " datasets.")
    count = 1
    student_instructions = []
    for data in student_instructions:
        # For here, we want to work with the input as context.
        my_robert.clear_context()
        # the output of a student instruction dataset is a question for Rob
        robs_answer = my_robert.get_response(data['output'])

        dataset = {
            'instruction': data['output'],
            'output': robs_answer,
            'input': '',
            'context': data['context'],
            'model': model_name,
            'type': 'instruction',
            'rating': 0,
            'comment': '',
            "isRated": False
        }
        db.get_database()['rateable_test_datasets'].insert_one(dataset)
        count = count + 1
        # Decomment the two lines below if you dont want a new line in the console.
        sys.stdout.write('Done with ' + str(count) + ' datasets. ')

    # Do dialogs as well here!
    print("Doing dialogs now")
    student_dialogs = db.get_filtered_student_dialogs(50)
    print("Going through " + str(len(student_dialogs)) + " datasets.")
    count = 1
    for data in student_dialogs:
        history = data['context'].split('\n')
        instruction = history[len(history) - 1]
        context = history[:-1]
        # For here, we want to work with the input as context.
        my_robert.set_context(context)
        # the output of a student instruction dataset is a question for Rob
        robs_answer = my_robert.get_response(instruction)

        dataset = {
            'instruction': instruction,
            'output': robs_answer,
            'input': data['context'],
            'context': '',
            'model': model_name,
            'type': 'dialog',
            'rating': 0,
            'comment': '',
            "isRated": False
        }
        db.get_database()['rateable_test_datasets'].insert_one(dataset)
        count = count + 1
        # Decomment the two lines below if you dont want a new line in the console.
        sys.stdout.write('Done with ' + str(count) + ' datasets. ')


def start_test_pipeline():
    '''
    Starts a test pipeline by testing the given robert models with
    various prompts, dialogs and questions.
    '''
    # We go through each model and test them
    tries = 3
    to_test = [m for m in test_models if m['test'] is True]
    print(str(datetime.now()))
    print("===================== Starting a new pipeline =====================")
    print("For that, we have " + str(len(to_test)) + " models to test.\n\n")
    for model in to_test:
        try:
            if(model['name'] in done_models):
                continue
            my_robert = robert(finetuned_path=build_finetuned_path(model['name']),
                               context_amount=4, dtype="bfloat16")
            print("Doing " + model['name'] + " now:")

            if(include_rouge):
                test_instruction_following_capabilities(model['name'], my_robert)
                test_dialog_capabilities(model['name'], my_robert)
            if(generate_rateable_datasets):
                build_rateable_dataset(model['name'], my_robert)

            print("Done with " + model['name'] + "!\n")
            done_models.append(model['name'])
            # Free the gpu from the model
            my_robert = ""
            gc.collect()
            torch.cuda.empty_cache()
            # I hope this gives pytorch enough time to free the memory. Otherwise, we crash here.
            time.sleep(5)
        except Exception as ex:
            print("Caught en exception")
            print(ex)
            # We want to try again if an error occured because it could be just
            # missing memory.
            if(tries > 0):
                print("Retrying again in 10 seconds.")
                time.sleep(10)
                tries = tries - 1
                start_test_pipeline()

    print(str(datetime.now()))
    print("===================== Done with the pipeline =====================")


prompt_text = '''
Rob knows the following:
[CONTEXT]

A student asked Rob, a virtual reality assistant:
[QUESTION]

Rob answered with:
[ANSWER]

Rob should only answer when he truly knows the answer. Otherwise he should excuse himself.
Rate Robs answer with a number from 1 to 10. Rate harshly!
Print only the number.'''


dialog_prompt_text = '''
A student is having a conversation with Rob, the virtual reality assistant. This is the chat history:
[HISTORY]

Rob knows the following:
[CONTEXT]

Rob continued the dialog with:
[ANSWER]

Rate Robs answer with a number from 1 to 10. Focus heavily on whether the answer has correct information given Robs knowledge! If the answer is false, give at max 3 points! If the answer is nonsensical give 1 point!
Print only the number.
'''


def start_chatgpt_pipeline():
    '''ChatGPT will give each answer a score.'''
    scores = db.get_rouge_scores(99999)
    count = 0
    for score in scores:
        test_model = [m for m in test_models if m['name'] == score['model']][0]
        if(test_model['test'] is False):
            print("Skipping for " + test_model['name'])
            continue
        answer = ''
        if(score['inp'] == ''):
            # time.sleep(0.5)
            # This is an instruction following test
            prompt = prompt_text.replace('[QUESTION]', score['instruction'])
            prompt = prompt.replace('[CONTEXT]', "\n".join(score['context'].split('[ITEM]')))
            prompt = prompt.replace('[ANSWER]', score['prediction'])
            print(prompt)
            answer = chatgpt_model.get_response(prompt).strip().replace("\n", "")
        else:
            # This is a dialog test
            prompt = dialog_prompt_text.replace('[HISTORY]', score['inp'])
            prompt = prompt.replace('[CONTEXT]', "\n".join(score['context'].split('[ITEM]')))
            prompt = prompt.replace('[ANSWER]', score['prediction'])
            print(prompt)
            answer = chatgpt_model.get_response(prompt).strip().replace("\n", "")
        try:
            s = int(answer)
            print("========================")
            print("Score: " + str(s))
            print("========================")
            db.insert_chatgpt_score(s, score['model'], score['instruction'],
                                    score['prediction'], score['inp'],
                                    score['context'], score)
        except Exception as ex:
            print(ex)
            print("Couldn't convert to int: " + answer)
        count = count + 1
        print("Done with " + str(count))


def start_student_instruction_generation():
    '''Creates X amount of new instructions by a student for robert'''
    params = get_parameters()
    my_student = robert(finetuned_path=build_finetuned_path("student_24k_para"),
                        is_student=True, context_amount=4)
    for i in range(100):
        context = random.sample(params, random.randint(1, 2))
        my_student.set_context(context)
        # The response of the student model is an instruction for Rob
        answer = my_student.get_response(student_instruction)
        dataset = {
            "instruction": student_instruction,
            "output": answer,
            "context": "[ITEM]".join(context),
            "model": "student_24k_para"
        }
        db.get_database()['student_instructions'].insert_one(dataset)


def start_student_dialog_generation():
    '''Creates X amount of new instructions by a student for robert'''
    params = get_parameters()
    my_student = robert(finetuned_path=build_finetuned_path("student_24k_para"),
                        is_student=True, context_amount=4, dtype="bfloat16")
    for i in range(200):
        # context here is the chat history. We have none for now.
        context = random.sample(params, random.randint(1, 2))
        history = []
        my_student.set_context(context)
        # The response of the student model is an instruction for Rob
        answer = my_student.get_response(student_instruction)
        history.append("Student: " + answer)
        dataset = {
            "instruction": student_dialog,
            "output": answer,
            "context": "\n".join(history),
            "model": "student_24k_para",
            "last_turn": "Student",
            "turns": 1
        }
        db.get_database()['student_dialogs'].insert_one(dataset)


def continue_student_dialog_generation():
    turn = 4
    dialogs = db.get_student_dialogs_by_turn(9999, turn)
    print("Found " + str(len(dialogs)) + " dialogs of turn " + str(turn))
    last_turn = dialogs[0]["last_turn"]
    print("Last turn was: " + str(last_turn))
    my_model = ''
    speaker = ''
    if(last_turn == "Student"):
        print("Initing Robert as our model")
        my_model = robert(finetuned_path=build_finetuned_path("robert_21k_chat_only_para"),
                          context_amount=2, dtype="bfloat16")
        speaker = "Rob"
    else:
        print("Initing the student as our model")
        my_model = robert(finetuned_path=build_finetuned_path("student_22k_chat_para"),
                          is_student=True, context_amount=2, dtype="bfloat16")
        speaker = "Student"
    # Now through each dialog, continue it.
    for dialog in dialogs:
        history = dialog['context'].split('\n')
        # If we have a history, then pass in the chat history
        # Robert doesnt take the last instruction into the context
        if(last_turn == "Student"):
            my_model.set_context(dialog['context'].split('\n')[:-1])
        else:
            my_model.set_context(dialog['context'].split('\n'))

        # The student gets the default prompt
        prompt = student_dialog
        # Rob gets the last question of the student as the input
        if(last_turn == "Student"):
            prompt = dialog["output"]

        answer = my_model.get_response(prompt)
        history.append(speaker + ": " + answer)
        dataset = {
            "instruction": prompt,
            "output": answer,
            "context": "\n".join(history),
            "model": "robert_21k_chat_only_para/student_22k_chat_para",
            "last_turn": speaker,
            "context_size": 2,
            "turns": turn + 1
        }
        db.get_database()['student_dialogs'].insert_one(dataset)


if __name__ == "__main__":
    db.init()
    print("Database initiated.")
    chatgpt_model.init(openai)
    print("Chatgpt initiated.")

    start_test_pipeline()
    # start_chatgpt_pipeline()
    if(generate_student_instructions):
        start_student_instruction_generation()
    if(generate_student_dialogs):
        # start_student_dialog_generation()
        continue_student_dialog_generation()
