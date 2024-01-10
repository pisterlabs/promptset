from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
import re
from components import transform


def load_cls_chain(model, tables):
  cls_chain = (
    PromptTemplate.from_template(
        f"""Given the user question below, classify it as either being about data {', '.join(tables)}""" + """
                                        
    Do not respond with more than one word.

    <question>
    {question}
    </question>

    Classification:"""
        )
        | model
        | StrOutputParser()
    )
  
  return cls_chain


def transform_output(res, cur,question, model):
    res, agg,table_name = res
    print(res, agg, table_name)

    if agg == None:
        results = transform.decimal_to_float(res)
        table_name = table_name.replace('"', "")
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'public';")
        
        col_name = cur.fetchall()
        
        print(col_name)
        # results = [{col_n[0] : v for col_n,v in zip(col_name,res)} for res in results]

        res = [" | ".join([f"{col_n[0]} : {v}" for col_n,v in zip(col_name,res)]) for res in results]
        print(res)
    
    # results = llm(f"""Generate final response based on the following question and the answer\n\nQUESTION:\n{query}\n\nANSWER:\n{agg}:{' '.join([str(x[0]) for x in res])}""")
    # print(results)
    # s = "\n"
    answer = '\n'.join([str(x[0]) for x in res]) 
    prompt = PromptTemplate.from_template("Generate final response based on the following question and the answer\n\nQUESTION:\n{query}\n\n"+f"ANSWER:\n{answer}")

    # print("ok jjo")

    return (prompt | model).invoke({"query":question, "answer":answer})

    # return answer

def gen_q(res):
    return res

# def gen_res(model, question):




def load_model_chain(model, sql_model, conn):
    cur = conn.cursor()

    sql_template = "Question: {sql_question}\nTable: {table_schema}\nSQL:"
    sql_prompt = PromptTemplate.from_template(sql_template)

    gen_prompt = PromptTemplate.from_template("Generate final response based on the following question and the answer\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}")

  
    full_chain = ( 
        {
            "table_schema":lambda x : find_table_schema(cur, x['question'], model), 
            "sql_question" : lambda x: x['question']
        } 
        | sql_prompt 
        | sql_model 
        | StrOutputParser()
        | {
            
            "out" : lambda x: parse_or_fix(conn, model, x, itemgetter('sql_question'), itemgetter('table_schema')),# "table_name":lambda x: [tab for tab in x['tables'].lower().split(' ') if tab in x['topic'].lower()][0]
            # "question" : lambda x: itemgetter('question')
            # "question_last": lambda x: itemgetter('question'),# | RunnableLambda(gen_q),
            
            }
        # | {"answer" : lambda x:transform_output(x['out'], cur, itemgetter('question'),model)}
        # | RunnableLambda(lambda x:x['answer'])
        ## | {"answer": lambda x:transform_output(x['out'], cur)}
        ## | gen_prompt
        ## | {"result" : model.invoke({"question":itemgetter('question')})}
        ## | RunnableLambda(lambda x:x['result'])
        ## | StrOutputParser()
    )
    
    return full_chain

def find_table_schema(cur, question, llm):

    cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")

    tables = [a[0] for a in cur.fetchall() if 'knowled' not in a[0] and 'pg_' not in a[0] and '_embedd' not in a[0] and '_cache' not in a[0]]

    cls_model = load_cls_chain(llm, tables)

    table_name = cls_model.invoke({'question':question}).split(" ")[-1].lower().replace(".","").replace(",","")
    # table_name = re.sub(r'[^a-zA-Z0-9]', '',cls_model.invoke({'question':question}).split(" ")[-1])
    # cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)

    print("TABLE NAME :",table_name)

    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'public';")
    col, d  = zip(*cur.fetchall())
    d = [x if x[:2] == 'te' or x[:2] =='da' else 'real' for x in d]
    
    col_names = ", ".join([f'"{a}" {b}' for a,b in zip(col,d)])

    print(col_names)
    
    return f"""{table_name} ( {col_names} )"""

# def query_sql(x, cur):
    



def parse_or_fix(conn, llm, text, question, table):#, question:str, table:str):
    cur = conn.cursor()
    # pattern = re.compile(r'\bFROM\s+(\w+)')
    # print(PromptTemplate.from_template("{question} | {table}"))
    text = "SELECT" + text.split(";")[0].replace('`','').split("SELECT")[-1]
    
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Question: {question}\n\n"
            "SQL table schema: {table}\n\n"
            "Based on the above table schema and question, fix the following SQL query:\n\n```text\n{input}\n```\nError: {error}"
            " Don't narrate, just respond with the fixed data."
            " Change the table name according to the above SQL table schema."
            " If and only if there is no aggregation function such as ['AVG', 'COUNT', 'SUM', 'MIN', 'MAX'], change the SQL query to select all columns"
            # " Don't change the after WHERE statement"
        )
        | llm
        | StrOutputParser()
    )

    e_ = None
    sql_agg = ['AVG', 'COUNT', 'SUM'] 
    for _ in range(4):
        # text = pattern.sub(f'FROM {str(table).split(" (")[0]}', text)
        try:
            
            # print(table_name)
            print("Fixed Text : ", text)

            agg = None 
            for x in sql_agg:
                if x in text:
                    agg = x
                    break
            topic = None
            if agg == None:
                pattern = r'"(.*?)"'
                try :
                    topic = re.search(pattern, text).group(0)
                    text = text.replace(topic, "*", 1)
                    topic = re.search(pattern, text).group(0)
                except :
                    topic = re.search(r'FROM(.*?)WHERE', text, re.IGNORECASE).group(0).strip().split(" ")[1]
                # text = transform.replace_aggregation_functions(text, table_name)[-1]
            print(text)
            
            
            # print(text)

            cur.execute(text)

            out = cur.fetchall()
            # print(out)
            # if len(out) == 0:
            #     raise Exception("The result is empty! Try other way, for example try to change the string value")
            # out = "\n".join([" | ".join([str(a) for a in x]) for x in out])
            # print(out)
            # print(type(out))
            return out,agg,topic#,question
        except Exception as e:
            conn.commit()
            text = fixing_chain.invoke({"input": text, "error": e, "question":question, "table":table})
            e_ = e
    raise e_




def load_model_chain_large(model, sql_model, conn):
    cur = conn.cursor()

    sql_template = "Generate SQL query based on the following Question and Table\nQuestion: {sql_question}\nTable: {table_schema}\nSQL:"
    sql_prompt = PromptTemplate.from_template(sql_template)

    gen_prompt = PromptTemplate.from_template("Generate final response based on the following question and the answer\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}")

    full_chain = ( 
        {
            "table_schema":lambda x : find_table_schema(cur, x['question'], model), 
            "sql_question" : lambda x: x['question']
        } 
        | sql_prompt 
        | sql_model 
        | StrOutputParser()
        | {
            
            "out" : lambda x: parse_or_fix(conn, model, x, itemgetter('sql_question'), itemgetter('table_schema')),# "table_name":lambda x: [tab for tab in x['tables'].lower().split(' ') if tab in x['topic'].lower()][0]
            # "question" : lambda x: itemgetter('question')
            # "question_last": lambda x: itemgetter('question'),# | RunnableLambda(gen_q),
            
            }
        # | {"answer" : lambda x:transform_output(x['out'], cur, itemgetter('question'),model)}
        # | RunnableLambda(lambda x:x['answer'])
        ## | {"answer": lambda x:transform_output(x['out'], cur)}
        ## | gen_prompt
        ## | {"result" : model.invoke({"question":itemgetter('question')})}
        ## | RunnableLambda(lambda x:x['result'])
        ## | StrOutputParser()
    )
    
    return full_chain