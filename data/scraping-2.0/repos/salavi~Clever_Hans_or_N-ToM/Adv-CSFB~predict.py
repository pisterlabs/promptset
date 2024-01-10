import argparse, time
import openai
import json
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unexpected_contents')
    parser.add_argument('--openai_api_key', default=None, type=str, required=True, help="API key to use GPT-3.")
    parser.add_argument('--model',
                        default=None,
                        type=str,
                        required=True,
                        help="list of available models: list of models text_davinci_001, "
                             "text_davinci_002, text_davinci_003, gpt-3.5-turbo-0301, gpt-4")
    parser.add_argument('--predict', action='store_true', help="Requests openai to generate responses")
    args = parser.parse_args()
    opt = vars(args)
    openai.api_key = args.openai_api_key
    input_path = f"data/{opt['dataset']}.jsonl"
    continue_index = 0
    if opt['predict']:
        predict(input_path, continue_index, opt)

    accuracy(opt)
    type_accuracy(opt)


def type_accuracy(opt):
    if opt['dataset'] == 'unexpected_transfer' or opt['dataset'] == 'unexpected_contents':
        types_results = {}
        with open(f'results_log/{opt["dataset"]}_{opt["model"]}_kosinski-test.txt') as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                fields = json.loads(line)
                temp = types_results.setdefault(fields['type'], ([], []))
                temp[0].append([fields[fields['truth']], fields[fields['belief']], fields[fields['belief']]])
                temp[1].append([fields['prediction1'], fields['prediction2'], fields['prediction3']])
                types_results[fields['type']] = temp

        for key, value in types_results.items():

            gold = np.array(value[0])
            predictions = np.array(value[1])
            accuracy_q1 = accuracy_score(gold[:, 0], predictions[:, 0])
            accuracy_q2 = accuracy_score(gold[:, 1], predictions[:, 1])
            accuracy_q3 = accuracy_score(gold[:, 2], predictions[:, 2])
            total_accuracy = (accuracy_q1 + accuracy_q2 + accuracy_q3) / 3
            joint_accuracy = 0
            for i in range(gold.shape[0]):
                joint_accuracy += int(accuracy_score(gold[i], predictions[i]))
            joint_accuracy /= gold.shape[0]
            print(key)
            print(len(value[0]))
            print(f"q1 accuracy: {accuracy_q1: .3f}")
            print(f"q2 accuracy: {accuracy_q2: .3f}")
            print(f"q3 accuracy: {accuracy_q3: .3f}")
            print(f"Average accuracy: {total_accuracy: .3f}")
            print(f"Joint `accuracy: {joint_accuracy: .3f}")

    # delighted question
    # elif opt['dataset'] == 'unexpected_contents':
    #     types_results = {}
    #     with open(f'output/{opt["dataset"]}_{opt["model"]}_kosinski.txt') as f_in:
    #         for i, line in enumerate(tqdm.tqdm(f_in)):
    #             fields = json.loads(line)
    #             temp = types_results.setdefault(fields['type'], ([], []))
    #             # print(fields[fields['belief']] + "     " + fields['prediction1'])
    #             temp[0].append([fields[fields['belief']]])
    #             temp[1].append([fields['prediction1']])
    #             types_results[fields['type']] = temp
    #
    #     for key, value in types_results.items():
    #         print(key)
    #         print(len(value[0]))
    #         gold = np.array(value[0])
    #         predictions = np.array(value[1])
    #         accuracy_q1 = accuracy_score(gold[:, 0], predictions[:, 0])
    #         print(f"q1 accuracy: {accuracy_q1: .3f}")


def accuracy(opt):
    gold = []
    predictions = []
    if opt['dataset'] == 'unexpected_transfer' or opt['dataset'] == 'unexpected_contents':
        with open(f'results_log/{opt["dataset"]}_{opt["model"]}_kosinski-test.txt') as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                fields = json.loads(line)
                gold.append([fields[fields['truth']], fields[fields['belief']], fields[fields['belief']]])

                predictions.append([fields['prediction1'], fields['prediction2'], fields['prediction3']])
        gold = np.array(gold)
        predictions = np.array(predictions)
        accuracy_q1 = accuracy_score(gold[:, 0], predictions[:, 0])
        accuracy_q2 = accuracy_score(gold[:, 1], predictions[:, 1])
        accuracy_q3 = accuracy_score(gold[:, 2], predictions[:, 2])
        total_accuracy = (accuracy_q1 + accuracy_q2 + accuracy_q3) / 3
        joint_accuracy = 0
        for i in range(gold.shape[0]):
            joint_accuracy += int(accuracy_score(gold[i], predictions[i]))
        joint_accuracy /= gold.shape[0]

        print(f"q1 accuracy: {accuracy_q1: .3f}")
        print(f"q2 accuracy: {accuracy_q2: .3f}")
        print(f"q3 accuracy: {accuracy_q3: .3f}")
        print(f"Average accuracy: {total_accuracy: .3f}")
        print(f"Joint `accuracy: {joint_accuracy: .3f}")
    #delighted one question
    # elif opt['dataset'] == 'unexpected_contents':
    #     with open(f'output/{opt["dataset"]}_{opt["model"]}_kosinski.txt') as f_in:
    #         for i, line in enumerate(tqdm.tqdm(f_in)):
    #             fields = json.loads(line)
    #             # print(fields[fields['belief']] + "     " + fields['prediction1'])
    #             gold.append([fields[fields['belief']]])
    #             predictions.append([fields['prediction1']])
    #     gold = np.array(gold)
    #     predictions = np.array(predictions)
    #     accuracy_q1 = accuracy_score(gold[:, 0], predictions[:, 0])
    #     print(f"q1 accuracy: {accuracy_q1: .3f}")


def predict(input_path, continue_index, opt):
    preprompt = get_preprompt(opt)
    unexpected_content_source = load_data('data/unexpected_contents_source.json')[0]['data']
    with open(f"results_log/{opt['dataset']}_{opt['model']}_kosinski-test_reordered_choices.txt", "a") as f_out:
        with open(input_path) as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                if i < continue_index:
                    continue
                print(i)
                fields = json.loads(line)
                if opt['dataset'] == 'unexpected_contents':
                    fields.update(unexpected_content_source[fields['idx']-1])
                prompts = get_prompt(preprompt, opt, fields, i, method1=True)
                for i, prompt in enumerate(prompts):
                    if opt['model'] == 'gpt-3.5-turbo-0301' or opt['model'] == 'gpt-4':
                        fields['prediction' + str(i+1)] = open_ai_chatgpt_request(opt['model'], prompt, i, 0).strip()
                    else:
                        fields['prediction' + str(i+1)] = open_ai_finalanswer_request(opt['model'], prompt, i, 0).strip()
                fields['prompts'] = prompts
                f_out.write(json.dumps(fields) + "\n")


def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))

    return data


def get_preprompt(opt):
    prompt = ""
    if opt['dataset'] == 'unexpected_contents' or opt['dataset'] == 'unexpected_transfer':
        with open(f"data/{opt['dataset']}_preprompt.txt") as f_in:
            for line in f_in:
                data = json.loads(line)
                prompt += 'Context: ' + data['context'] + ' ' + data['question'] + '\nQuestion: Fill in the blank with the best option.' \
                              + '\n' + f"- {data['o1']}\n" + f"- {data['o2']}\n" + \
                          f"Answer: {data['label']}" + "\n\n"
    return prompt


def get_prompt(preprompt, opt, fields, sample_count ,method1=False):
    prompts = []
    if sample_count % 2 == 0:
        choices = '- ' + fields['o1'].strip() + '\n' + '- ' + fields['o2'].strip()
    else:
        choices = '- ' + fields['o2'].strip() + '\n' + '- ' + fields['o1'].strip()

    if opt['dataset'] == 'unexpected_transfer':
        context = fields['txt'].strip()
        for i in range(1, 4):
            question =  fields['q' + str(i)].strip() + " _\n" + 'Question: Fill in the blank with the best option.'
            prompt = preprompt + "Context: " + context + '\n' + question + '\n' + choices + '\n' + "Answer:"
            prompts.append(prompt)
    if opt['dataset'] == 'unexpected_contents':
        if not method1:
            context = fields['txt'].strip().replace("[o1]", fields['o1'])
            context = context.replace("[o2]", fields['o2'])
            context = context.replace("[ctr]", fields['ctr'])
            context += fields['xpro'].capitalize() + ' is delighted to have found this ' + fields['ctr'] + '.'
            question = 'Question: Fill in the blank with the best option. ' + fields['xpro'].capitalize() + ' ' + fields['q3'].strip() + " _"
            prompt = preprompt + "Context: " + context + '\n' + question + '\n' + choices + '\n' + "Answer:"
            prompts.append(prompt)
        else:
            #kosinski questions
            context = fields['txt'].strip().replace("[o1]", fields['o1'])
            context = context.replace("[o2]", fields['o2'])
            context = context.replace("[ctr]", fields['ctr'])
            question1 = fields['xpro'].capitalize() + \
                        ' opens the ' + fields['ctr'] + ' and looks inside. ' + fields['xpro'].capitalize() + \
                        ' can clearly see that it is full of _\n' + 'Question: Fill in the blank with the best option. '
            question2 = fields['xpro'].capitalize() + \
                        ' believes that it is full of _\n' + 'Question: Fill in the blank with the best option.'
            question3 =  fields['xpro'].capitalize() + \
                        f' calls {fields["pos_pro"]} friend to tell them that he has just found a {fields["ctr"]} full of _\n' + 'Question: Fill in the blank with the best option.'
            prompts = [preprompt + "Context: " + context + '\n' + question1 + '\n' + choices + '\n' + "Answer:",
                       preprompt + "Context: " + context + '\n' + question2 + '\n' + choices + '\n' + "Answer:",
                       preprompt + "Context: " + context + '\n' + question3 + '\n' + choices + '\n' + "Answer:"]

    return prompts


def open_ai_chatgpt_request(model, prompt, i, counter):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=30,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['message']['content'].strip()
    except:
        if counter < 3:
            time.sleep(10)
            return open_ai_chatgpt_request(model, prompt, i, counter + 1)
        else:
            print(prompt)
            print("continue from:" + str(i))
            exit()


def open_ai_finalanswer_request(model, prompt, i, counter):
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=30,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=['\n', '.']
        )
        return response['choices'][0]['text'].strip()
    except:
        if counter < 3:
            time.sleep(10)
            return open_ai_finalanswer_request(model, prompt, i, counter + 1)
        else:
            print(prompt)
            print("continue from:" + str(i))
            exit()


if __name__ == '__main__':
    main()
