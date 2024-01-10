import pandas as pd
import jellyfish
from data.questions_const import *
from data.utils import load_perovskite_data
from datetime import datetime

FT_FLOAT = 'float'
FT_SEQ_SUFFIX = "_seq"
FT_STRING = 'string'
FT_INT = 'int'
FT_DATE = 'date'
FT_BOOLEAN = 'boolean'

DB_PATH = "../dataset/questions/questions_db.csv"

def format_questions_and_save_5_4(
        protocol_path='Extraction protocolls version 5_4.xlsx',
        sheet_name='Master',
        output_path='questions_5_4.txt',
        save_output=True,
        print_output=False):
    df = pd.read_excel(protocol_path, sheet_name=sheet_name)
    questions = df.iloc[:, 0].to_list()
    for i, question in enumerate(questions):
        new_question = questions[i]
        answer_format = ''
        if print_output: print(new_question)
        new_question = "What is the " + questions[i].replace(".", "")
        if "[" in new_question:
            splits = new_question.split(" [")
            answer_format = splits[1][:-1]
            new_question = splits[0] + "? Format of answer: [" + \
                           answer_format + "]"
        if df.iloc[i, 1] == df.iloc[i, 1]:
            # for some unsolvable issue the answer is converted from 0,
            # 1 to False,True. So additional conversion is needed
            example_answer = str(df.iloc[i, 1])
            if answer_format and "TRUE" not in answer_format:
                example_answer = example_answer. \
                    replace("False", "0"). \
                    replace("True", "1")
            new_question += ", Example:" + '"' + example_answer + '"'
        questions[i] = new_question
        if print_output: print(questions[i] + "\n")
    # save the questions to a txt file delimited each question by a "\n"
    if save_output:
        with open(output_path, "w") as output_path:
            for question in questions:
                output_path.write(str(question) + '\n')
    return questions


def load_questions_from_txt(questions_path):
    questions = []
    with open(questions_path, "r") as questions_path:
        for line in questions_path:
            questions.append(line.strip())
    return questions


def question_to_field(question: str):
    return best_5p_question_to_field.get(question)


def infer_field_from_question(question: str) -> (int, str):
    pervo_df = pd.read_csv(r"C:\Users\Raz_Z\Projects\perovskite-miner"
                           r"\Perovskite_database_content_all_data.csv",
                           low_memory=False)
    df_fields = pervo_df.columns
    for i, field in enumerate(df_fields):
        if jellyfish.jaro_distance(field, question) > 0.8:
            print("question: ", question)
            print("field: ", field)
            return i, field


def counted_tokens_data(questions_df : pd.DataFrame) -> pd.DataFrame:
    from apis.gpt_api import openai_count_tokens as count_tokens
    pervo_df = load_perovskite_data()
    tokens_per_field = pervo_df.applymap(lambda x : count_tokens(str(
        x))).mean().rename_axis(FIELD_NAME).reset_index().round(1)
    merged = pd.merge(questions_df, tokens_per_field, on=FIELD_NAME,
                      how='left').rename(columns={0: TOKENS_PER_ANSWER})
    merged[TOKENS_PER_QUESTIONS] = merged[GPT_QUESTION].apply(
        lambda x: count_tokens(x))
    return merged


def create_questions_db(output_path: str) -> None:
    protocol_path = r'../dataset/questions/Extraction protocolls version 5_4.xlsx'
    sheet_name = 'Master'
    df = pd.read_excel(protocol_path, sheet_name=sheet_name)
    df.reset_index(inplace=True)
    df['questions'] = format_questions_and_save_5_4(
        protocol_path=protocol_path,
        save_output=False,
        print_output=False)
    df.columns = [QID, PROTOCOL_QUESTION, EXAMPLE_ANSWER,GPT_QUESTION]
    df[FIELD_NAME] = df[PROTOCOL_QUESTION].apply(lambda question: question_to_field(question))
    df = df[[QID, PROTOCOL_QUESTION, FIELD_NAME, GPT_QUESTION, EXAMPLE_ANSWER]]
    df = counted_tokens_data(df)
    df = add_question_type(df)
    df = add_is_kpi_field(df)
    df = add_question_score_column(df)
    df.to_csv(output_path, index=False)


def add_question_score_column(df: pd.DataFrame) -> pd.DataFrame:
    df[QUESTION_SCORE] = df[PROTOCOL_QUESTION].apply(
        lambda x: kpi_question_scores[x] if x in kpi_question_scores.keys() else None
    )
    return df


def is_date(string: str) -> bool:
    try:
        datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False


def infer_question_type(example_answer: str) -> str:
    example_answer = str(example_answer)
    if example_answer == '':
        return ''
    if '|' in example_answer:
        first_part_in_seq = example_answer.split('|')[0].strip()
        return infer_question_type(first_part_in_seq)+ FT_SEQ_SUFFIX
    elif 'TRUE' in example_answer.upper() or 'FALSE' in example_answer.upper():
        return FT_BOOLEAN
    elif is_date(example_answer):
        return FT_DATE
    elif example_answer.replace(".", "").isdigit():
        if '.' in example_answer:
            return FT_FLOAT
        return FT_INT
    else:
        return FT_STRING


def add_question_type(df: pd.DataFrame) -> pd.DataFrame:
    df[QUESTION_TYPE] = df[EXAMPLE_ANSWER].apply(lambda x: infer_question_type(x))
    return df


def add_is_kpi_field(df: pd.DataFrame) -> pd.DataFrame:
    df[IS_KPI_FIELD] = df[FIELD_NAME].apply(lambda x: x is not None)
    return df


# Todo: add the doc prioritization score as a new column
# Todo: remove the example answer on binary fields
# Todo: loading numbers such as 1 and 0 is still converted to True and False

if __name__ == '__main__':
    create_questions_db(output_path=DB_PATH)
    print("done")
