from datetime import datetime
from config import Config
from flask import request
from flask_jwt_extended import create_access_token, get_jwt, get_jwt_identity, jwt_required
from flask_restful import Resource
from mysql_connection import get_connection
from mysql.connector import Error
from email_validator import validate_email, EmailNotValidError
from utils import check_password, hash_password
import openai

# chat-gpt
class ConsultationResource(Resource) :
    # 고민 상담 API ( 질문과 응답을 DB에 저장 )
    @jwt_required()
    def post(self) :
    
        # 상담 기능 

        userId = get_jwt_identity()
        data = request.get_json()
        content = data["question"]
        type = data["type"]
        
        openai.api_key = Config.openAIKey
        # 유능하고 친절한 고민상담가
        if type == 0:
            system_message = "You are a competent and kind trouble counselor who listens to people's concerns and provides helpful advice."
        # 객관적이고 냉철한 고민상담가
        elif type == 1:
            system_message = "You are an objective and cool-headed trouble counselor who listens to people's concerns and provides rational advice."
        # 편안한 친구같은 고민상담가
        else:  # counselor_type == 2
            system_message = "You are a comforting friend-like trouble counselor who listens to people's concerns and provides warm and supportive advice."

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": content+" 한국어 210자 정도로 답변해줘. "}
            ]
        )

        print(completion.choices[0].message['content'].strip())

        response_message = completion.choices[0].message['content'].strip()

        # DB에 저장
        try : 
            connection = get_connection()
            connection.begin()
            query = '''insert into consultation
                    (userId,question,answer,type)
                    values
                    (%s,%s,%s,%s)'''
            record = (userId,content,response_message,type)
            cursor = connection.cursor()
            cursor.execute(query,record)
            connection.commit()


        except Error as e :
            connection.rollback()
            print(e)
            return{'error':str(e)},500
        finally:
            cursor.close()
            connection.close()
        
        return {'result':'success','question' : content,'answer': response_message },200


# 질문 히스토리 가져오기

class ConsultationHistoryResource(Resource):

    @jwt_required()

    def get(self) :

        userId = get_jwt_identity()

        try :
            connection = get_connection()

            query = '''select * from consultation
                    where userId=%s
                    order by createdAt desc;'''
            

            cursor = connection.cursor(dictionary=True)
            
            record = (userId,)

            cursor.execute(query,record)
            
            resultList = cursor.fetchall()

            print(resultList)

            i = 0
            for row in resultList :
                resultList[i]['createdAt']=row['createdAt'].isoformat()
                i = i+1


        except Error as e:
            print(e)


            return{"result" : "fail", "error" : str(e)}, 500
        finally:
            cursor.close()
            connection.close()
        
        return {"result" : "success", "result" : resultList,"count":len(resultList)} ,200
    


# 히스토리 삭제
class DeleteHistoryResource(Resource):

    @jwt_required()

    def delete(self, id) :

        userId = get_jwt_identity()


        try :
            connection = get_connection()
            connection.begin()

            query = '''delete from consultation
                    where userId = %s and id=%s;'''
            

            cursor = connection.cursor()
            
            record = (userId,id)

            cursor.execute(query,record)
            
            connection.commit()
            



        except Error as e:
            connection.rollback()
            print(e)
            return{"result" : "fail", "error" : str(e)}, 500
        finally:
            cursor.close()
            connection.close()
        
        return {"result" : "success"} ,200
    
# 검색 기능   
class ConsultationSearchResource(Resource):
    @jwt_required()
    def get(self):
        userId = get_jwt_identity()
        keyword = request.args.get('keyword')

        try:
            connection = get_connection()

            query = '''SELECT * FROM consultation
                       WHERE userId = %s AND (question LIKE %s OR answer LIKE %s)
                       ORDER BY createdAt DESC;'''

            keyword_pattern = f'%{keyword}%'

            cursor = connection.cursor(dictionary=True)
            record=(userId, keyword_pattern, keyword_pattern)
            cursor.execute(query, record)

            search_results = cursor.fetchall()

            for idx, row in enumerate(search_results):
                search_results[idx]['createdAt'] = row['createdAt'].isoformat()

        except Error as e:
            print(e)
            return {"result": "fail", "error": str(e)}, 500
        finally:
            cursor.close()
            connection.close()

        return {"result": "success", "searchResults": search_results, "count": len(search_results)}, 200








            




