import os

import numpy as np
import openai
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Use GPT-3 to label Bible verses (identify semantic domains) by answering pre-generated y/n questions

# model_name = "ada:ft-personal:sd-labeler-8-2023-04-17-14-23-46"
### model_name = "davinci:ft-personal:sd-labeler-8-2023-04-17-15-12-52"
# model_name = "ada:ft-personal:sd-labeler-9-2023-04-18-16-06-58"
### model_name = "davinci:ft-personal:sd-labeler-9-2023-04-18-14-52-28"
# model_name = "ada:ft-personal:sd-labeler-10-2023-04-19-11-55-29"
# model_name = "ada:ft-personal:sd-labeler-11-2023-04-19-15-08-24"
# model_name = "ada:ft-personal:sd-labeler-11-2023-04-19-16-10-35" # actually the 12th version
# model_name = "ada:ft-personal:sd-labeler-13-2023-04-20-21-21-04"
# model_name = "ada:ft-personal:sd-labeler-14-2023-04-21-11-25-33"
# model_name = "ada:ft-personal:sd-labeler-16-2023-04-24-11-38-16"

# model_name = "ada:ft-personal:sd-labeler-15-2023-04-21-13-31-05" # {7: {'accuracy': 0.87, 'F1': 0.8670076726342711}, 8: {'accuracy': 0.84, 'F1': 0.830220713073005}, 9: {'accuracy': 0.79, 'F1': 0.7851662404092072}, 10: {'accuracy': 0.82, 'F1': 0.8125}, 11: {'accuracy': 0.81, 'F1': 0.7973333333333333}, 12: {'accuracy': 0.77, 'F1': 0.7631551848419318}}
# model_name = "babbage:ft-personal:sd-labeler-15-2023-04-25-14-17-41"
model_name = "curie:ft-personal:sd-labeler-15-2023-04-24-12-46-47"                  # {7: {'accuracy': 0.91, 'F1': 0.907928388746803}, 8: {'accuracy': 0.86, 'F1': 0.8440285204991087}, 9: {'accuracy': 0.83, 'F1': 0.8236331569664903}, 10: {'accuracy': 0.83, 'F1': 0.8186666666666667}, 11: {'accuracy': 0.84, 'F1': 0.8283998283998284}, 12: {'accuracy': 0.85, 'F1': 0.841621792841305}}
### model_name = "davinci:ft-personal:sd-labeler-15-2023-04-24-12-23-09"

# model_name = "babbage:ft-personal:sd-labeler-17-2023-04-27-16-21-59"
# model_name = "curie:ft-personal:sd-labeler-17-2023-04-25-12-48-17"

# model_name = "babbage:ft-personal:sd-labeler-18-2023-04-27-16-52-12"

answered_question_count = None
total_question_count = None


def answer_question(question):
    if question[-1] != '?':
        print(f"Error: question does not end with '?': '{question}'")
        return f"Error: question does not end with '?': '{question}'"
    question = question[:-1] + ' ->'

    # # randomly select questions
    # if random.randint(0, 19) != 0:
    #     answered_question_count += 1
    #     return None

    # question = lowercase_question(question)  # for Ada-12/13

    openai.api_key = os.environ['OPENAI_API_KEY']
    try:
        # answer = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     temperature=0.0,
        #     request_timeout=5,
        #     messages=[
        #         # {"role": "system", "content": "Only answer with y or n."},
        #         {"role": "system",
        #          "content": 'Answer all questions with "ok". Only if they are absurd, answer with "absurd". Remember that all quotes are from the four gospels.'},
        #         {"role": "user", "content": question},
        #     ]
        # )["choices"][0]["message"]["content"]
        answer = openai.Completion.create(
            model=model_name,
            temperature=0,
            request_timeout=5,
            max_tokens=1,
            prompt=f"{question}",
        )["choices"][0]["text"]

        if answer in (' 1', ' yes'):
            answer = 1
        elif answer in (' 0', ' no'):
            answer = 0
        else:
            print(f"Error: answer is not ' yes' or ' no': '{answer}'")
    except Exception as e:
        print(f"Error: {e}")
        answer = str(e)

    # print progress
    global answered_question_count
    answered_question_count += 1
    if answered_question_count % 10 == 0:
        print(f"{answered_question_count}/{total_question_count} questions answered")
        print(f"{question} {answer}")

    return answer


def answer_questions(selected_df, result_column_name):
    global total_question_count
    total_question_count = len(selected_df)
    selected_df[result_column_name] = selected_df['direct_question'].apply(answer_question)

    # filter out questions that were already answered (answer is not 0 or 1)
    # repeat 3 times for questions that were not answered (e.g., because of timeout)
    df_unanswered = selected_df[selected_df[result_column_name].isin((0, 1, "0", "1")) == False]
    total_question_count = len(df_unanswered)
    df_unanswered[result_column_name] = df_unanswered['direct_question'].apply(answer_question)
    selected_df.update(df_unanswered)

    df_unanswered = selected_df[selected_df[result_column_name].isin((0, 1, "0", "1")) == False]
    total_question_count = len(df_unanswered)
    df_unanswered[result_column_name] = df_unanswered['direct_question'].apply(answer_question)
    selected_df.update(df_unanswered)

    df_unanswered = selected_df[selected_df[result_column_name].isin((0, 1, "0", "1")) == False]
    total_question_count = len(df_unanswered)
    df_unanswered[result_column_name] = df_unanswered['direct_question'].apply(answer_question)
    selected_df.update(df_unanswered)
    return selected_df


def label_questions(df, is_evaluatable=False, test_file_num=None):
    # df['answer'] = df['answer'].astype(int)
    # df = df[~df['qid'].str.startswith('8')]

    # # select rows for semantic domain
    # df = df[df['qid'].str.startswith('3')]
    # df = df.reset_index(drop=True)

    # print number of 0 and 1 answers
    print(df['answer'].value_counts())

    # # load data/2_sd_labeling/ada_answers_in.csv into a dataframe
    # df_answers = pd.read_csv("data/2_sd_labeling/ada_answers_in.csv")
    # assert (len(df) == len(df_answers))
    # df[result_column_name] = df_answers[result_column_name]

    # # randomly select questions with equal distribution of 0 and 1 answers
    # seed = int(time.time())
    # num_questions = 100
    # df = df[df['answer'] == 0]\
    #     .sample(n=num_questions, random_state=seed)\
    #     .append(df[df['answer'] == 1].sample(n=num_questions, random_state=seed))

    global answered_question_count
    answered_question_count = 0
    result_column_name = 'gpt3_answer'

    if is_evaluatable:
        df_answered = df[df['answer'].isna() == False]
        df_answered = answer_questions(df_answered, result_column_name)
        df[result_column_name] = np.nan
        df.update(df_answered)

        # evaluate accuracy, precision, recall, F1 score, and ROC AUC
        df[result_column_name] = df[result_column_name].astype(int)
        df['correct'] = df[result_column_name] == df['answer']
        f1 = f1_score(df["answer"], df[result_column_name], average="macro")
        precision = precision_score(df["answer"], df[result_column_name], average="macro")
        recall = recall_score(df["answer"], df[result_column_name], average="macro")
        accuracy = df['correct'].mean()
        roc_auc = roc_auc_score(df["answer"], df[result_column_name])
        print(
            f'test file: {test_file_num}, F1: {f1:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, accuracy: {accuracy:.2f}, roc_auc: {roc_auc:.2f}')

        # put question column at the end
        df = df[['correct', result_column_name, 'answer', 'direct_question']]

        # # print entire dataframe with incorrect answers
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_colwidth', None)
        # print(df[df['correct'] == False])

        # save results
        df.to_csv(f"data/2_sd_labeling/test sets/answers_{test_file_num}_F1:_{f1}_{model_name}.csv", index=False)

        return {'accuracy': accuracy, 'F1': f1}
    else:
        df_unanswered = df[df['answer'].isna() == True]
        df_unanswered = answer_questions(df_unanswered, result_column_name)
        df[result_column_name] = np.nan
        df.update(df_unanswered)
        df.to_csv(f"data/2_sd_labeling/matched_questions_answered_curie_4.csv", index=False)
        return None

    # 100 --> 0.79, 0.81
    # 200 --> 0.865
    # 50x0 + 50x1 --> 0.78
    # full file: 200x0 + 200x1 --> 0.8025
    # SD 1: 200x0 + 200x1 --> 0.8225, 0.805, 0.825, 0.815, 0.79, 0.785
    # SD 1 with rephrased questions: 0.79, 0.7475, 0.81, 0.75, 0.79, 0.85, 0.845, 0.81

    # SD 1 with rephrased questions: 0.89, 0.875, 0.87
    # SD 1 without rephrased questions: 0.905, 0.9


def test_model(test_file_num):
    df = pd.read_csv(f"data/2_sd_labeling/test sets/test_set_{test_file_num}.csv",
                     usecols=["direct_question", "answer", "qid"])
    return label_questions(df, True, test_file_num)


def answer_unanswered_questions():
    df = pd.read_excel("data/2_sd_labeling/matched_questions.xlsx",
                      usecols=["direct_question", "answer", "qid"], nrows=156501)
    df = df.iloc[137199:]
    return label_questions(df)


if __name__ == "__main__":
    # # test model
    # test_results = dict()
    # for test_file_num in [10, 11, 12]:
    #     test_results[test_file_num] = test_model(test_file_num)
    # print(test_results)
    # print(f"average F1 score: {np.mean([test_results[test_file_num]['F1'] for test_file_num in test_results]):.2f}")
    # print(
    #     f"average accuracy: {np.mean([test_results[test_file_num]['accuracy'] for test_file_num in test_results]):.2f}")

   answer_unanswered_questions()
