import sys 
sys.path.append(".")
from answer.SQL_skeleton.llm import LLM
import pandas as pd
import argparse
import time
from multiprocessing import Queue,Process
from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from databricks import sql
import os
import json
import re
import itertools
import numpy as np
from dotenv import load_dotenv, find_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class compare_query():
    def __init__(self):
        _ = load_dotenv(find_dotenv())

        self.DBWS_HOST = os.getenv("dbws_host_domain")
        self.DBWS_HTTP_PATH = os.getenv("dbws_host_path")
        self.DBWS_PAT = os.getenv("dbws_pat")

    def execute_sql_query(self,query):
        result_rows = []

        with sql.connect(server_hostname = self.DBWS_HOST,
                        http_path        = self.DBWS_HTTP_PATH,
                        access_token     = self.DBWS_PAT) as conn:

            with conn.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()

                for row in result:
                    result_rows.append(row)

        if len(result_rows) == 1:
            return result_rows[0]
        
        return result_rows

    def get_prompt_template_for_query_2(self):
        prompt_template = PromptTemplate.from_template(
        """you are a SQL smart programmer.\
        convert the given sql query {query} in human readable format.\
        convert the query in simple human readable format so that non a human with non programming background can also understand, with full  explaination in at most 50 words.\
        NOTE: COUNT(*) and COUNT(event_id) are the same things.\
        NOTE: Remember SQL FUNCTION AVG (column)=SUM(column)/length(column).\
        Note: Case touches : total DISTINCT case_id_value 
        note: COUNT(column) and COUNT(DISTINCT column) are different things
        
        """
        )

        return prompt_template
    def get_pompt_template_for_equivalent(self):
        prompt_template = PromptTemplate.from_template(
        """you are a programmer.\
        find weather these two statement {statement_1} and {statement_2} are same or not.\
        JUST COMPARE THE RESULTS.\
        Dont compare the methods of statements.\
        Points to be keep in mind while compairing the two statements.\
        1.Ignore the time range/month/year in which these two statements are compared.\
        2.condition agent_type !=0 and agent_type>0 are same in all the senerios.\
        3.Highest Case touches : highest total Unique case_id_value  
        
        
        If both statements  are doing the same thing  then only return '@@' and give reason why in not more than 20 words.\
        If both statements  are not doing the same thing then only return '$$' and give reason why in not more than 20 words.\
        
        """
        )

        return prompt_template

    def getting_llm_response(self,query,tmp):
        llm_model=LLM(20,4)
        prompt_template=self.get_prompt_template_for_query_2()
        prompt=prompt_template.format(query=query)
        result=llm_model.get_response(prompt,tmp)
        result=result.replace('content=','')
        result=result.replace('additional_kwargs={} example=False','')
        return result


    def get_llm_response_for_equivalancy(self,result_1,result_2):
        llm_model=LLM(20,4)
        prompt_template=self.get_pompt_template_for_equivalent()
        prompt=prompt_template.format(statement_1=result_1,statement_2=result_2)
        response=llm_model.get_response(prompt,0)
        response=response.replace('content=','')
        response=response.replace('additional_kwargs={} example=False','')
        return response

    def DATASET_generalize_according_to_deltalake_database(self,query):
        query=query.replace('"','')
        query = re.sub(' +', ' ', query)
        # query=query.replace("'","")
        current_query=query.split(' ')

        modified_query=''
        index=0
        
        while(index<len(current_query)):
            if current_query[index]=='FROM' or current_query[index]=='from':

                next_word=current_query[index+1]
                if next_word[0]=='(':
                    modified_query+='FROM '

                else:
                    modified_query+='FROM unum_askskan.events_delta_tb ' 
                    index+=1
            else:
                modified_query+=current_query[index]+' '
            index+=1

        return str(modified_query)
        
    def modify_query_for_evaluating_sql_answer(self,query):
        #removing not supported function in deltalake
        for fun in ['strftime','STRFTIME']:
            query=query.replace(fun+"('%Y-%m', event_time) = '2023-04'"," event_date >= '2023-04-01' AND event_date <= '2023-04-30' ")
            query=query.replace(fun+"('%w', event_time)"," DATE_FORMAT(event_date, 'EEEE') ")
            query=query.replace(fun+"('%H', event_time)"," HOUR(event_time)")
        
        return query
    def modify_query_for_evaluating_equivalancy_condition(self,query):
        #removing not supported function in deltalake
        for fun in ['strftime','STRFTIME']:
            query=query.replace(fun+"('%Y-%m', event_time) = '2023-04'"," event_date >= '2023-04-01' AND event_date <= '2023-04-30' ")
            query=query.replace(fun+"('%w', event_time)"," DATE_FORMAT(event_date, 'EEEE') ")
            query=query.replace(fun+"('%H', event_time)"," HOUR(event_time)")
        
        #removing asc desc limit 1 ,limit 10 , limit 1
        flag=1
        for fun1 in ['ASC ','asc ','DESC ','desc ','']:
            for fun2 in ['limit ','LIMIT ']:
                for num  in [str(i) for i in range(1,11)]:
                    if fun1+fun2+num in query and flag:
                        query=query.replace(fun1+fun2+num,fun1+fun2+'1000')
                        flag=0
                
        
        
        return query

    def modify_evaluated_sql_answer(self,answer):
        size = np.array(answer).shape
        if len(size)==1:
            for index2 in range(0,len(answer)):
                if isinstance(answer[index2],int) or isinstance(answer[index2],float):
                    answer[index2]=round(float(answer[index2]),2)
            for index2 in range(0,len(answer)):
                if isinstance(answer[index2],str):
                    if answer[index2].replace('.','',1).isdigit():
                        answer[index2]=round(float(answer[index2]),2)
                    else:
                        answer[index2]=answer[index2].replace('"','')
            return answer
            
        for index1 in range(0,len(answer)):
            for index2 in range(0,len(answer[index1])):
                if isinstance(answer[index1][index2],int) or isinstance(answer[index1][index2],float):
                    answer[index1][index2]=round(float(answer[index1][index2]),2)
        
        for index1 in range(0,len(answer)):
            for index2 in range(0,len(answer[index1])):
                if isinstance(answer[index1][index2],str):
                    if answer[index1][index2].replace('.','',1).isdigit():
                        answer[index1][index2]=round(float(answer[index1][index2]),2)
                    else:
                        answer[index1][index2]=answer[index1][index2].replace('"','')
        return answer

    def get_equivalancy_result_without_union(self,rows_1,rows_2,count_1):
        sz_1=len(np.array(rows_1).shape)
        sz_2=len(np.array(rows_2).shape)
        #not equivalent
        if sz_1!=sz_2:
            return 0
        rows_1=self.modify_evaluated_sql_answer(rows_1)
        rows_2=self.modify_evaluated_sql_answer(rows_2)
        # print(rows_1)
        # print()
        # print(rows_2)
        # print(sz_1)
        if sz_1>1:
            for comb in itertools.permutations([i for i in range(0,len(rows_2[0]))],sz_1):
                total_distinct_rows=rows_1
                
                for row_id in range(0,len(rows_2)):
                    curr=[]
                    for index in comb:
                        curr.append(rows_2[row_id][index])
                    
                    total_distinct_rows.append(curr)
                df = pd.DataFrame(total_distinct_rows)
                df1=df.drop_duplicates()
                if count_1==df1.shape[0]:
                    return 'Equivalent'
            return 'Not Equivalent'

        for comb in itertools.permutations([i for i in range(1,len(rows_2))],sz_1):
            total_distinct_rows=rows_1
            
            for index in comb:
                total_distinct_rows.append(rows_2[index])
            
            df = pd.DataFrame(total_distinct_rows)
            df1=df.drop_duplicates()
            if count_1==df1.shape[0]:
                return 'Equivalent'
        return 'Not Equivalent'
        
    def get_result(self,query_1,query_2):

        
        

        query_1=self.DATASET_generalize_according_to_deltalake_database(query_1)
        query_2=self.DATASET_generalize_according_to_deltalake_database(query_2)
        try:
            x=self.execute_sql_query(self.modify_query_for_evaluating_sql_answer(query_2))
            x=self.execute_sql_query(self.modify_query_for_evaluating_sql_answer(query_1))
        except: 
            return "Not Equivalent",'invalid query'
        #evaluating sql query answer and compairing
        #evaluating sql query answer and compairing
        # print("yes")
        evaluated_answer_of_sql_query_1=self.modify_evaluated_sql_answer(json.loads((json.dumps(self.execute_sql_query(self.modify_query_for_evaluating_sql_answer(query_1)), indent=2))))
        evaluated_answer_of_sql_query_2=self.modify_evaluated_sql_answer(json.loads((json.dumps(self.execute_sql_query(self.modify_query_for_evaluating_sql_answer(query_2)), indent=2))))
        
        length_of_column_in_query_1=len(evaluated_answer_of_sql_query_1)
        length_of_column_in_query_2=len(evaluated_answer_of_sql_query_2)
        
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        # similarity_between_evaluated_sql_query_1_and_query_2=get_evaluated_sql_result_max_similarity(evaluated_answer_of_sql_query_1,evaluated_answer_of_sql_query_2)
        similarity_between_evaluated_sql_answers= cosine_similarity(model.encode([' '.join([str(elem) for elem in evaluated_answer_of_sql_query_1]),' '.join([str(elem) for elem in evaluated_answer_of_sql_query_2])]))[0][1]
    
        
        # using llm model to find equivalency
        result_1=self.getting_llm_response(query_1,0)        
       

        
        itr=0
        temperature=[0]
        fully_equivalent=False
        E_C_N_condition=''#reasoning given by llm model
        E_C_N_condition_reason=''

        llm_model_reasoning_similarity=0 # similarity in query by llm model
        llm_model_reasoning=''#reasoning for similarity given by llm model
        # print(result_1)
        while(itr<len(temperature)):

            #query 2 result    
            result_2=self.getting_llm_response(query_2,temperature[itr])
            # print(result_2)
            
            #similarity between two results
            embeddings = model.encode([result_1,result_2])
            similarity_matrix=cosine_similarity(embeddings)
            llm_model_reasoning_similarity=max(llm_model_reasoning_similarity,similarity_matrix[0][1])


            #checking statement equivalancy
            # response=get_llm_response_for_equivalancy(result_1,result_2)
            response=self.get_llm_response_for_equivalancy(result_1,result_2)
            
            # print(response)
            if '@@' in response:
                fully_equivalent=True
                llm_model_reasoning=response.replace('@@','')
                break
            else:
                llm_model_reasoning=response.replace('$$','')

        
            itr+=1
            time.sleep(1)
        flag=1
        if fully_equivalent and llm_model_reasoning_similarity>=0.75 and flag:
            flag=0
            E_C_N_condition='E'
            E_C_N_condition_reason='similarity between two queries : '+str(llm_model_reasoning_similarity)+' AND '+llm_model_reasoning
            
        if llm_model_reasoning_similarity>=0.98 and flag:
            flag=0
            E_C_N_condition='E'
            E_C_N_condition_reason='similarity between two queries : '+str(llm_model_reasoning_similarity)
            
        if llm_model_reasoning_similarity<0.98 and llm_model_reasoning_similarity>=0.5 and flag:
            flag=0
            E_C_N_condition='C'
            E_C_N_condition_reason='similarity between two queries : '+str(llm_model_reasoning_similarity)
        
        if flag:
            E_C_N_condition='N'
            E_C_N_condition_reason='similarity between two queries is : '+str(llm_model_reasoning_similarity)+ ' AND '+llm_model_reasoning
        
        

        if (E_C_N_condition=='E' or E_C_N_condition=='C')and similarity_between_evaluated_sql_answers>=0.99:
            return 'Equivalent','['+str(E_C_N_condition)+','+llm_model_reasoning+','+'evaluated_answer_similarity - '+str(similarity_between_evaluated_sql_answers)+']'
            
        if E_C_N_condition=='N':
            return 'Not Equivalent','['+str(E_C_N_condition)+','+llm_model_reasoning+']'
        

        query_1=self.modify_query_for_evaluating_equivalancy_condition(query_1)
        query_2=self.modify_query_for_evaluating_equivalancy_condition(query_2)
        
        count_1=json.loads(json.dumps(self.execute_sql_query("SELECT COUNT(*)  FROM ("+query_1+") AS count1")))[0]
        count_2=json.loads(json.dumps(self.execute_sql_query("SELECT COUNT(*)  FROM ("+query_2+") AS count1")))[0]
        # print("yes")
        if count_1!=count_2:
            return 'Not Equivalent','['+str(E_C_N_condition)+','+'columns counts of queries not equal'+','+'evaluated_answer_similarity - '+str(similarity_between_evaluated_sql_answers)+']'

    
        if length_of_column_in_query_1==length_of_column_in_query_2:
            #in most cases
            count_3=json.loads(json.dumps(self.execute_sql_query("SELECT COUNT(*)  FROM (SELECT * FROM ("+query_1+") UNION SELECT * FROM ("+query_2+")) AS unioned")))[0]
            if count_1==count_3:
                return 'Equivalent','['+str(E_C_N_condition)+','+llm_model_reasoning+','+'evaluated_answer_similarity - '+str(similarity_between_evaluated_sql_answers)+']'
            else:
                return 'Not Equivalent','['+str(E_C_N_condition)+','+llm_model_reasoning+','+'evaluated_answer_similarity - '+str(similarity_between_evaluated_sql_answers)+']'
            
        
        rows_1=json.loads(json.dumps(self.execute_sql_query("SELECT *  FROM ("+query_1+") AS count1"),indent=2))
        rows_2=json.loads(json.dumps(self.execute_sql_query("SELECT *  FROM ("+query_2+") AS count1"),indent=2))
        return self.get_equivalancy_result_without_union(rows_1,rows_2,count_1),'['+str(E_C_N_condition)+','+llm_model_reasoning+','+'evaluated_answer_similarity - '+str(similarity_between_evaluated_sql_answers)+']'

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
            '-i',nargs='?' ,help='path of input question csv',type=str, default='/Users/mohd.a/Desktop/repo/Prototypes/final.csv'
            )
        parser.add_argument(
            '-o',nargs='?' ,help='path of output question csv',type=str, default='not_provided'
            )
        
        
        args = parser.parse_args()
        return [int(args.f),int(args.l),str(args.c1),str(args.c2),str(args.i),str(args.o)]    
        
        

if __name__=="__main__":
    [starting_index,ending_index,query_column_name_1,query_column_name_2,file_path_of_input_question_csv,file_path_of_export_question_csv]=get_args()

    question_csv=pd.read_csv(file_path_of_input_question_csv)
    export_csv=question_csv.copy()
    
    #final result E or N 
    final_result_column=['not_evaluated' for ele in range(0,len(export_csv))]
    final_result_column_name='final result - ('+query_column_name_1+','+query_column_name_2+')'

    #reason
    Reason_column=['not_evaluated' for ele in range(0,len(export_csv))]
    Reason_column_name='Reason - ('+query_column_name_1+','+query_column_name_2+')'

    
    

    # restoring previous results from csv if present
    if final_result_column_name in export_csv:
        for index in range(0,export_csv.shape[0]):
            final_result_column[index]=export_csv.loc[index,final_result_column_name]
            Reason_column[index]=export_csv.loc[index,Reason_column_name]
            
    
    #model for compairing two sentence
    
    #llm model
    llm_model=LLM(20,4)
    for index in range(0,len(export_csv)):
        start=time.time()
        id=export_csv['#'][index]

        #dont want to compare these queries for these
        if id<starting_index or id>ending_index or export_csv['deprecated'][index]=='yes':
            continue

        #multiple interpretation
        if export_csv['Single Interpretation'][index]=='no':
            #for now
            final_result_column[index]='Equivalent'
            Reason_column[index]='Multiple Interpretation'
            continue
        
        # print(id)
        compare_queries=compare_query()
        final_result_column[index],Reason_column[index]=compare_queries.get_result(export_csv[query_column_name_1][index],export_csv[query_column_name_2][index])
        


    export_csv[final_result_column_name]=  final_result_column
    export_csv[Reason_column_name]=Reason_column
    if file_path_of_export_question_csv!='not_provided':
        export_csv.to_csv(file_path_of_export_question_csv,index=False,header=True)
    else:
        export_csv.to_csv(file_path_of_input_question_csv,index=False,header=True)


        

        