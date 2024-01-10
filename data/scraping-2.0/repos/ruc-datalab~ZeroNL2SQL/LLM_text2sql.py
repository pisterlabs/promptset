import os
import openai
import sqlite3
import argparse
from tqdm import tqdm
import time
import re
from value_match import value_match
from get_colval_map import get_colval_map
import subprocess
import json


def execute(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        exec_result = cursor.fetchall()
    except (sqlite3.OperationalError, Exception) as e:
        return e, None
    return '', exec_result


def find_invalid_functions(sql):
    pattern = r"[A-Z]+\(.*\)"
    valid_funcs = ['AVG', 'MAX', 'SUM', 'COUNT', 'MIN']
    Q = re.findall(pattern=pattern, string=sql)
    funcs = []
    while Q:
        cur = Q.pop(0)
        left = 0
        while cur[left] != '(':
            left += 1
        if cur[:left] not in valid_funcs:
            funcs.append(cur[:left])
        Q.extend(re.findall(pattern=pattern, string=cur[left+1:-1]))
    return funcs


def normalize(sql):
    pre_len = len(sql)
    while 1:
        sql = sql.replace('\n', ' ')
        sql = sql.replace('\r', ' ')
        sql = sql.replace('  ', ' ')
        sql = sql.replace('<>', '!=')
        if len(sql) == pre_len:
            break
        else:
            pre_len = len(sql)
    if sql[:6].lower() != 'select':
        if "```" in sql:
            start_index = sql.index("```") + 3
            end_index = start_index + sql[start_index:].index("```")
            return sql[start_index:end_index]
        try:
            start_index = sql.index('SELECT')
        except:
            start_index = 0
        if start_index == 0:
            try:
                start_index = sql.index('select')
            except:
                start_index = 0
                print("normalize error:", sql)
        return sql[start_index:]
    return sql


Table_hint = 'Requirement: The SQL query must consist of these tables: {table_list}.\n'
Structure_hint = 'Requirement: The SQL query must be in this format: {structure}.\n'
instruction = 'Note that the content in [] after the column name are the values in that column. Use values that actually exist in the database, which are important for proper SQL execution.\n'
keywords = 'the only functions you can use are: [max(), min(), count(), sum(), avg()], Do not use other functions, or the SQL will not execute correctly.\nSometimes commonsense knowledge is required to correctly understand the user question. Don\'t be careless!\nDo not alias tables in the SQL query.\n'
db_id_str = "### The database contains the following tables:\n# "
table_sep = "\n# "

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', dest='key', type=str, help="openai api kei")
    parser.add_argument('--test_set_name', dest='test_set_name', type=str, help="")
    parser.add_argument('--value_match_method', default='sbert',
                        help='use fuzzy|bert|sbert|word2vec')
    args = parser.parse_args()
    openai.api_key = args.key
    if args.test_set_name == "kaggledbqa":
        input_path = "data/kaggledbqa/test_with_template.json"
        db_path = "data/kaggledbqa/databases"
        output_path = f"data/kaggledbqa/prediction.json"
    elif args.test_set_name[:2] == "DB":
        input_path = f"data/drspider/{args.test_set_name}/questions_post_perturbation.json"
        db_path = f"data/drspider/{args.test_set_name}/database_post_perturbation"    
        output_path = f"data/drspider/{args.test_set_name}/prediction.json"      
    elif args.test_set_name[:3] in ["NLQ", "SQL"]:
        input_path = f"data/drspider/{args.test_set_name}/questions_post_perturbation.json"
        db_path = f"data/drspider/{args.test_set_name}/databases"
        output_path = f"data/drspider/{args.test_set_name}/prediction.json"

    token_cnt = [0, 0]

    try:
        with open(output_path) as f:
            results = json.load(f)
    except FileNotFoundError as e:
        results = []

    begin = len(results)
    with open(input_path) as f:
        test_data = json.load(f)
    while begin < len(test_data):
        print(f'begin={begin}')
        for item in tqdm(test_data[begin:]):
            db_id = item['db_id']
            db = os.path.join(db_path, db_id, db_id + '.sqlite')
            message_tracks = []
            for candidate_table_list in item["pred_table_list"]:
                # table schema
                table_list = candidate_table_list.split(' ')
                table_schema_list = list()
                for table in item["tables"]:
                    if table in table_list:
                        table_schema_list.append(item["tables"][table])
                db_schema = db_id_str + table_sep.join(table_schema_list)
                table_str = ', '.join(table_list)

                # input to LLM
                if table_str == "":
                    continue
                input_str = db_schema + '\n' + instruction + item['question'] + Structure_hint.format(structure=item['pred_structure'].lower()) + Table_hint.format(table_list=table_str) + keywords + f"Translate the question into a SQL query that starts with \"{item['pred_select']} from\". Only show the SQL query.\n"
                message_tracks.append({"messages": [{"role": "user", "content": input_str}], "pred_sql": None, "exec_result": None})

            item['prediction'] = []
            try:
                for track_id in range(len(message_tracks)):
                    print(f"-------------track {track_id}---gold table list = {item['gold_table_list']}------------------")
                    print(message_tracks[track_id]["messages"][0]["content"])
                    LLM_return = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", n=1,
                        stream=False, temperature=0.0, top_p=1.0,
                        frequency_penalty=0.0, presence_penalty=0.0,
                        messages=message_tracks[track_id]["messages"],
                    )
                    output = LLM_return.choices[0].message['content']
                    message_tracks[track_id]["messages"].append({"role": "system", "content": output})
                    token_cnt[0] += LLM_return["usage"]["total_tokens"]
                    # extract SQL from the return value
                    pred_sql = normalize(output)
                    print(f"pred sql: {pred_sql}")
                    message_tracks[track_id]["pred_sql"] = pred_sql

                    # check the validity of SQL
                    error, _ = execute(pred_sql, db)
                    if error != '':
                        try:
                            invalid_funcs = find_invalid_functions(pred_sql)
                        except:
                            invalid_funcs = []
                        if invalid_funcs:
                            print(f"invalid functions: {invalid_funcs}")
                            message_tracks[track_id]["messages"].append({"role": "user", "content": f"Error feedback: 1. {error} 2. Don't use {', '.join(invalid_funcs)} functions in the SQL query, you can only use the most basic functions. Please rewrite the SQL query to make it execute correctly. Only show the SQL query.\n"})
                        else:
                            message_tracks[track_id]["messages"].append({"role": "user", "content": f"Error feedback: 1. {error}. Please rewrite the SQL query to make it execute correctly. Only show the SQL query.\n"})
                        print(error)
                        LLM_return = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo", n=1,
                            stream=False, temperature=0.0, top_p=1.0,
                            frequency_penalty=0.0, presence_penalty=0.0,
                            messages=message_tracks[track_id]["messages"],
                        )
                        output = LLM_return.choices[0].message['content']
                        token_cnt[0] += LLM_return["usage"]["total_tokens"]
                        pred_sql = normalize(output)
                        print(f"pred sql: {pred_sql}")
                        message_tracks[track_id]["pred_sql"] = pred_sql
                        message_tracks[track_id]["messages"].append({"role": "system", "content": output})

                    # predicate calibration
                    feedback = ''
                    # extract predicate
                    pred_colval_map = get_colval_map(db, pred_sql)
                    print(f"pred_colval_map: {pred_colval_map}")
                    if 'like' in pred_sql.lower():
                        # skip LIKE
                        pred_colval_map = []
                    feedback_idx = 0
                    for tabcol, val in pred_colval_map:
                        if isinstance(val, str) and '.' in val:
                            continue
                        try:
                            tab, col = tabcol.split('.')
                        except:
                            print(f"parse colval error: {pred_colval_map}")
                            continue
                        if isinstance(val, str):
                            val = val.replace('\"', '')
                        (match_tabcol, match_val) = value_match(db, item['question'], item['label'], pred_sql, tab, col, val, match_method=args.value_match_method, k=3)
                        # value is wrong
                        if match_tabcol == tabcol and match_val != val and match_val is not None:
                            feedback_idx += 1
                            feedback += f'{str(feedback_idx)}. Column {tabcol} does not contain the value "{val}", but it contains a value "{match_val}" '
                        # column is wrong
                        elif match_tabcol != tabcol and match_val is not None:
                            feedback_idx += 1
                            feedback += f'{str(feedback_idx)}. Column {tabcol} does not contain value "{val}", but the column {match_tabcol} contains value "{match_val}"'

                    if feedback != '':
                        message_tracks[track_id]["messages"].append({"role": "user", "content": f"Database feedback: {feedback}. Please rewrite it, only show the SQL query.\n"})
                        print(feedback)
                        LLM_return = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo", n=1,
                            stream=False, temperature=0.0, top_p=1.0,
                            frequency_penalty=0.0, presence_penalty=0.0,
                            messages=message_tracks[track_id]["messages"],
                        )
                        output = LLM_return.choices[0].message['content']
                        token_cnt[0] += LLM_return["usage"]["total_tokens"]
                        pred_sql = normalize(output)
                        print(f"pred sql: {pred_sql}")
                        message_tracks[track_id]["pred_sql"] = pred_sql

                    # save execution results as a basis for selecting SQL
                    _, exec_result = execute(message_tracks[track_id]["pred_sql"], db)
                    message_tracks[track_id]["exec_result"] = exec_result
                    item['prediction'].append([message_tracks[track_id]["pred_sql"], message_tracks[track_id]["exec_result"]])
                    if exec_result is not None and len(exec_result) > 0:
                        break

                results.append(item)
                begin += 1
                token_cnt[1] += 1

            except Exception as e:
                print(e)
                break

        print("Stop : %s" % time.ctime())
        time.sleep(60)
        print("Begin : %s" % time.ctime())
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    pred_sql_path = 'test-suite-sql-eval/predict.txt'
    gold_sql_path = 'test-suite-sql-eval/gold.txt'
    if os.path.exists(pred_sql_path):
        os.remove(pred_sql_path)
    if os.path.exists(gold_sql_path):
        os.remove(gold_sql_path)

    with open(output_path, "r") as rf:
        data = json.load(rf)
        for idx, item in enumerate(data):
            # ideally, it should not happen
            if len(item["prediction"]) == 0:
                print("empty!!")
                with open(pred_sql_path, 'a') as af:
                    af.write("None\n")
            # if the first SQL has a suitable execution result,
            # or if there is only one sql,
            # use the second SQL as the prediction
            elif (item["prediction"][0][1] is not None and len(item["prediction"][0][1]) > 0) or len(item['prediction']) == 1 or item["prediction"][1][1] is None:
                with open(pred_sql_path, 'a') as af:
                    af.write(item['prediction'][0][0].replace("'", '"') + "\n")
            # if the first SQL does not have a suitable execution result, use the second SQL as the prediction
            else:
                with open(pred_sql_path, 'a') as af:
                    af.write(item['prediction'][1][0].replace("'", '"') + "\n")

            with open(gold_sql_path, 'a') as af:
                af.write(f'{item["label"]}\t{item["db_id"]}' + "\n")

    # get execution accuracy
    command = f'python test-suite-sql-eval/evaluation.py --gold {gold_sql_path} --pred {pred_sql_path} --etype exec --db {db_path}'
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    with open(f'eval.output', 'w') as f:
        f.write(output.decode("utf-8"))

    print(f"Average Token use for each sample: {token_cnt[0] / token_cnt[1]}")
