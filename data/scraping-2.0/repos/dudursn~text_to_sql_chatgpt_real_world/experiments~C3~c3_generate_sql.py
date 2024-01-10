import time
import re
from c3_consistency_output import get_sqls_self_consistency
from langchain.callbacks import get_openai_callback

def generate_sql(messages, llm, db, question, callback=None):
    
    results = []
    p_sqls = []
    for j in range(5):
       
        reply = generate(llm, messages, callback)
        p_sqls = reply
        temp = []
        for p_sql in p_sqls:
            p_sql = 'SELECT ' + p_sql
            p_sql = p_sql.replace("SELECT SELECT", "SELECT")
            try:
                p_sql = fix_select_column(p_sql)
            except:
                print(f"fix_select_column err, p_sql: {p_sql}")
                pass
            p_sql = p_sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
            p_sql = p_sql.replace("\n", " ")
            while "  " in p_sql:
                p_sql = p_sql.replace("  ", " ")
            temp.append(p_sql)
        p_sqls = temp
        if is_valid(p_sqls[0], db):
            break
        else:
            print(f're_id: {j} p_sql: {p_sqls[0]} exec error...')
            time.sleep(0.5)
            if j < 4:
                print(f'generate again')
    result = {}
    result['question'] = question
    result['p_sqls'] = []
    for sql in p_sqls:
        result['p_sqls'].append(sql)
    results.append(result)
    total = len(p_sqls)
    # time.sleep(1)
    p_sql_final = get_sqls_self_consistency(results, total, db)
    return p_sql_final

def generate(llm, messages, callback=None):
    
    reply = None
    attempts = 0
    while reply is None:
        try:
            with get_openai_callback() as cb:
                reply = llm.generate(messages)
                if callback is not None:
                    callback({"sql_generation":cb})
        except Exception as e:
            print(e)
            print(f"api error, wait for 3 seconds and retry...")
            time.sleep(3)
            attempts +=1
            if attempts > 4:
                print(f"Api error")
                break
            pass
            
    p_sqls = get_response(reply)
    return p_sqls

def get_response(responses):
    
    p_sqls = []
    for p_sql_response in responses.generations[0]:
        p_sql = p_sql_response.text
        p_sql = 'SELECT ' + p_sql
        p_sql = p_sql.replace("SELECT SELECT", "SELECT")
        try:
            p_sql = fix_select_column(p_sql)
        except:
            print(f"fix_select_column err, p_sql: {p_sql}")
            pass
        p_sql = p_sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
        p_sql = p_sql.replace("\n", " ")
        while "  " in p_sql:
            p_sql = p_sql.replace("  ", " ")
        p_sqls.append(p_sql)

    return p_sqls
    

def fix_select_column(sql):
    # sql = "SELECT DISTINCT model FROM cars_data JOIN car_names ON cars_data.id = car_names.makeid JOIN model_list ON car_names.model = model_list.model WHERE year > 1980;"
    sql = sql.replace("\n", " ")
    sql_list = sql.split("=")  # 给等号两边腾出空格
    sql = " = ".join(sql_list)
    while "  " in sql:
        sql = sql.replace("  ", " ")
    sql_tokens = sql.split(" ")
    select_ids = []
    from_ids = []
    join_ids = []
    eq_ids = []
    first_where_id = -1
    first_group_by_id = -1
    first_having_id = -1
    for id, token in enumerate(sql_tokens):
        if token.lower() == "select":
            select_ids.append(id)
        if token.lower() == "from":
            from_ids.append(id)
        if token.lower() == "join":
            join_ids.append(id)
        if token.lower() == "=":
            eq_ids.append(id)
        if token.lower() == "where" and first_where_id == -1:
            first_where_id = id
        if token.lower() == "group" and id < len(sql_tokens) - 1 and sql_tokens[id+1].lower() == "by" and first_group_by_id == -1:
            first_group_by_id = id
        if token.lower() == "having" and first_having_id == -1:
            first_having_id = id

    if len(eq_ids) == 0 or len(join_ids) == 0:
        return sql
    # assert len(select_ids) == len(from_ids)
    for i in range(len(select_ids[:1])):  ## 先只考虑最外层的select
        select_id = select_ids[i]
        from_id = from_ids[i]
        tmp_column_ids = [i for i in range(select_id + 1, from_id)]
        column_ids = []
        id = 0
        while id < len(tmp_column_ids):
            item = sql_tokens[id]
            if item.lower() == "as":
                id += 2
                continue
            column_ids.append(tmp_column_ids[id])
            id += 1
        column_table_mp = {}
        if i == len(select_ids) - 1:  # last select
            for j in range(len(join_ids)):
                if (first_where_id != -1 and join_ids[j] > first_where_id) or first_group_by_id != -1 and join_ids[j]:
                    break
                eq_id = eq_ids[j]
                left_id, right_id = eq_id - 1, eq_id + 1
                left_column, right_column = sql_tokens[left_id], sql_tokens[right_id]
                if "." not in left_column or "." not in right_column:
                    continue
                column_left = left_column.split(".")[1]
                column_right = right_column.split(".")[1]
                column_table_mp[column_left] = left_column
                column_table_mp[column_right] = right_column
        else:
            pass

        if len(column_table_mp) == 0:
            return sql
        for column_id in column_ids:
            column = sql_tokens[column_id]
            if "." not in column:
                if column in column_table_mp.keys():
                    sql_tokens[column_id] = column_table_mp[column]
                elif len(column) > 0 and column[-1] == "," and column[:-1] in column_table_mp.keys():
                    sql_tokens[column_id] = column_table_mp[column[:-1]] + ","

    recovered_sql = " ".join(sql_tokens)

    return recovered_sql

def exec_on_db_(query: str, db):
    try:
        result = db.run(query)
        return "result", result
    except Exception as e:
        return "exception", e
    

def is_valid(sql, db):
    flag, _ = exec_on_db_(sql, db)
    if flag == "exception":
        return 0
    else:
        return 1
