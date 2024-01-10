import json, os
from answerGenerator import generate_answers
from evaluationMetrics import llm_selfevaluation, exact_match
from loadDataset import LoadDataset
from openai import OpenAI
 
api_key="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"

all_data = LoadDataset.read_all_json_files('../results/questions/generated3/')
output_directory = ('../results/questions/generated4/')

for i, dataset in enumerate(all_data):
    if i <= 20:
        continue
    context = dataset['context']
    count_questions = dataset['count']
    yesno_questions = dataset['yesno']

    print("Context ", i)

    for idx, question_list in enumerate(dataset['count_variations']):
        gt_answer = dataset['count_gt'][idx]

        print("  Count ", idx)

        for dict_idx, question_dict in enumerate(question_list):
            print("    Dict idx ", dict_idx)
            question = question_dict['question']
            answer = generate_answers(context, question, api_key).lower()
            print('answer: ', answer)
            print('type: ', type(answer))
            llm_selfevaluation_out = llm_selfevaluation(context, gt_answer, answer, api_key)
            exact_match_output = exact_match(gt_answer, answer)
            question_dict['answer'] = answer
            question_dict['exact_match'] = exact_match_output
            question_dict['llm_selfevaluation'] = llm_selfevaluation_out
            # try:
            #     if exact_match_output == 'yes':
            #         question_dict['exact_match'] = 'yes'
            #     else:
            #         question_dict['exact_match'] = 'no'

            # except:
            #     print('cannot convert to int')
            #     question_dict['exact_match'] = 'N'

    for idx, question_list in enumerate(dataset['yesno_variations']):

        print("  Yesno ", idx)
        gt_answer = dataset['yesno_gt'][idx]

        for dict_idx, question_dict in enumerate(question_list):
            print("    Dict idx ", dict_idx)

            question = question_dict['question']
            answer = generate_answers(context, question, api_key).lower()
            print('answer: ', answer)
            print('type: ', type(answer))
            question_dict['answer'] = answer
            
            llm_selfevaluation_out = llm_selfevaluation(context, gt_answer, answer, api_key)
            exact_match_output = exact_match(gt_answer, answer)
            question_dict['exact_match'] = exact_match_output
            question_dict['llm_selfevaluation'] = llm_selfevaluation_out
            # try:
            #     # print('enter try')
            #     if float(exact_match_output) == 'yes':
            #         question_dict['exact_match'] = 'no'
            #     else:
            #         question_dict['exact_match'] = 'yes'

            # except:
            #     print('cannot convert to int')
            #     question_dict['exact_match'] = 'N'

        # print("Finish yesno")

    # Generate a filename for each dataset
    output_filename = f"context_{i}.json"
    output_path = os.path.join(output_directory, output_filename)

    # Save the dataset with generated questions to a JSON file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(dataset, json_file, ensure_ascii=False, indent=4)

    # print("Finish saving")
