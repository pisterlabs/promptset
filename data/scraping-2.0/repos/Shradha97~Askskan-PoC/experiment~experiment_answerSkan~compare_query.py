import sys 
sys.path.append(".")
from answer.test.llm import LLM
import pandas as pd
import argparse
import time
from multiprocessing import Queue,Process
from langchain import PromptTemplate
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',nargs='?',help='starting index', type=int,default=1
        )
    parser.add_argument(
        
        '-l',nargs='?' ,help='ending index',type=int, default=11000
        )

    parser.add_argument(
        '-c1',nargs='?',help='sql query column-1 name', type=str,required=True
        )
    parser.add_argument(
        '-c2',nargs='?' ,help='sql query column-2 name',type=str, required=True
        )
    parser.add_argument(
        '-i',nargs='?' ,help='path of input question csv',type=str, default='answer/in/questions.csv'
        )
    parser.add_argument(
        '-o',nargs='?' ,help='path of output question csv',type=str, default='not_provided'
        )
    
    
    args = parser.parse_args()
    return [int(args.f),int(args.l),str(args.c1),str(args.c2),str(args.i),str(args.o)]
def get_conditions():
    conditions = ["testing_query must do the same thing as correct_query does. i.e. testing_query is calculating the same variable as correct query correct_query does.",
                  "WHERE Clause Conditions in testing_query must exactly matches with WHERE Clause Conditions ,column names in correct_query.",
                  "Column names used in testing_query must exactly matches with Column names in correct_query.",
                  "testing_query use the same identical formula in correct query correct_query does",
                  "There must not be any differnece between correct_query and testing_query.",
                  "testing_query must produce exactly same result or variable output as correct_query.",
                  "Igonore ASC / DEC LIMIT Condition while compairing",
                  "Ignore whitespaces",
                  "Ignore the table/Dataset used while compairing"
                  "Ignore column_name is NOT NULL condition while compairing"
                  ""

    ]
    note=["Don't Assume any assumptions from your own.",
          "columns having integer type values can never attain negative values",
          "Ignore different syntaxes"
          "Don't compare SQL queries by SQL Aliases (temperory names given to column),instead compare by column name"
          
    ]       
    return conditions,note
def get_prompt_template():
    prompt_template = PromptTemplate.from_template(
    """you are a SQL smart programmer.\
       Given two SQL queries named correct_query {correct_query} and testing_query {testing_query}.\
       Your task is to check weather {testing_query} is FULLY EQUIVALENT to {correct_query}.\
       Here, FULLY EQUIVALENT means For any DATABASE ,{testing_query} will give exactly same results as {correct_query}.\
       First consider NOTE {note}.\
       for FULLY EQUIVALENT {testing_query} must satisfy the following {conditions} .\
       
       if all {conditions} are satisfied then return '@@' and reason.\
       else return '$$' and reason why they are not equivalent
     """
)
    return prompt_template
def getting_llm_response(query_1,query_2):
    llm_model=LLM(10,4)
    prompt_template=get_prompt_template()
    condition,note=get_conditions()
    prompt=prompt_template.format(correct_query=query_1,testing_query=query_2,conditions=condition,note=note)
    result=llm_model.get_response(prompt)
    return result
    
    


if __name__=="__main__":
    [starting_index,ending_index,query_column_name_1,query_column_name_2,file_path_of_input_question_csv,file_path_of_export_question_csv]=get_args()

    question_csv=pd.read_csv(file_path_of_input_question_csv)
    export_csv=question_csv.copy()
    result_column=['not_evaluated' for ele in range(0,len(export_csv))]
    reason_column=['not_evaluated' for ele in range(0,len(export_csv))]
    result_column_name='result '+query_column_name_1+' '+query_column_name_2
    reason_column_name='reason '+query_column_name_1+' '+query_column_name_2
    if result_column_name in export_csv:
        for index in range(0,export_csv.shape[0]):
            result_column[index]=export_csv.loc[index,result_column_name]
    if reason_column_name in export_csv:
        for index in range(0,export_csv.shape[0]):
            reason_column[index]=export_csv.loc[index,reason_column_name]
    
    
   
    for index in range(0,len(export_csv)):
        id=export_csv['#'][index]
        if id<starting_index or id>ending_index or export_csv['deprecated'][index]=='yes':
            if export_csv['deprecated'][index]=='yes':
                result_column[index]='Not_evaluated'
            continue
        
        print(id,end=" ")
        query_1=export_csv[query_column_name_1][index]
        query_1= query_1.replace("unum_askskan.events_delta_tb", "Dataset")
        query_2=export_csv[query_column_name_2][index]
        query_2= query_2.replace("unum_askskan.events_delta_tb", "Dataset")
        result=''

        result=getting_llm_response(query_1,query_2)
        
        result=result.replace('content=','')
        result=result.replace('additional_kwargs={} example=False','')
        print(result)
        if '@@' in result:
            result_column[index]='Equivalent'

            result.replace('@@','')
            result=result.replace('\n','')
            reason_column[index]=result.replace('@@','')
        else:
            result_column[index]='Not Equivalent'
            result=result.replace('\n','')
            reason_column[index]=str(result.replace('$$',''))
        time.sleep(1)


    
    export_csv[result_column_name]=result_column
    export_csv[reason_column_name]=reason_column
        
    if file_path_of_export_question_csv!='not_provided':
        export_csv.to_csv(file_path_of_export_question_csv,index=False,header=True)
    else:
        export_csv.to_csv(file_path_of_input_question_csv,index=False,header=True)


        