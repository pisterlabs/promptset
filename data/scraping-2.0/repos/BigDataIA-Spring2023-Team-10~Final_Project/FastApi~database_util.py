from dotenv import load_dotenv
import datetime
import pymysql
import os
load_dotenv()
import pinecone
import openai

class database_methods():
    
    ##function to connect to database 
    def connect_db(self):
        try:  
            conn = pymysql.connect(
                    host = os.getenv("host"), 
                    user = os.getenv("user"),
                    password = os.getenv("password"),
                    db = os.getenv("db"))
            cursor = conn.cursor()
            return conn,cursor
        except Exception as error:
            print("Failed to connect {}".format(error))
            
    ## Function to check if the user is still eligible to make api calls 
    def check_if_eligible(self,username):
        try:
            print(username)
            if username == "admin":
                return True
            else:
                _,cursor=self.connect_db()
                query="SELECT * FROM user_data WHERE username='{}'".format(username)
                cursor.execute(query)
                rows = cursor.fetchall()
                rows=rows[0]
                if rows[6] == None:
                    last_request_time=datetime.datetime.now()
                    self.update_last_req_time(username,datetime.datetime.now())
                    self.update_count_for_user(username,0)
                elif rows[6] != None:
                    last_request_time = datetime.datetime.strptime(str(rows[6]), '%Y-%m-%d %H:%M:%S')
                time_elapsed=datetime.datetime.now() - last_request_time
                if time_elapsed > datetime.timedelta(hours=1):
                    self.update_count_for_user(username,1)
                    return True
                elif time_elapsed < datetime.timedelta(hours=1):
                    allowed_count=self.get_allowed_count(rows[5])
                    if rows[7] == allowed_count:
                        return False
                    elif int(rows[7]) < allowed_count:
                        self.update_count_for_user(username,rows[7]+1)
                        return True
        except Exception as e:
            print("check_if_eligible: "+str(e))
            return "failed_insert"
    
    ## Function to update the last request time for the user
    def update_last_req_time(self,username,timestamp):
        try:
            conn,cursor=self.connect_db()
            query=f"UPDATE user_data SET user_last_request = '{timestamp}' where username='{username}'"
            cursor.execute(query)
            conn.commit()
            conn.close()
        except Exception as e:
            print("update_last_req_time: "+str(e))
            return "failed_insert"
       
    ## Function to update the count of api calls for the user
    def update_count_for_user(self,username,count):
        try:
            conn,cursor=self.connect_db()
            query = f"UPDATE user_data SET user_request_count = '{count}' WHERE username = '{username}'"
            cursor.execute(query)
            conn.commit()
            conn.close()
            return "password_updated"
        except Exception as e:
            print("update_count_for_user: "+str(e))
            return "failed_insert" 
        
    def get_allowed_count(self, tier):
            if tier == 'free':
                return 5
            if tier == 'gold':
                return 10
            if tier == 'platinum':
                return 15
    def generate_id(self):
        import random
        import string
        length = 25
        characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
        random_string = ''.join(random.choice(characters) for i in range(length))
        return random_string 

    def fetch_user(self,user_name):
        try:
            _,cursor=self.connect_db()
            query="SELECT * FROM user_data WHERE username='{}'".format(user_name)
            cursor.execute(query)
            rows = cursor.fetchall()
            if len(rows)==0:
                return "no_user_found"
            else:
                print(rows[0])
                return rows[0]
        except Exception as e:
            return 'Exception'   
    def add_user(self,username,password,restaurant_name,user_tier):
        try:
            conn,cursor=self.connect_db()
            sql_insert = "INSERT INTO user_data (username, password, restaurant_name, user_tier, restaurant_id) VALUES (%s, %s, %s, %s, %s)"
            restaurant_id = self.generate_id()
            record = (username, password, restaurant_name, user_tier, restaurant_id)
            cursor.execute(sql_insert,record)
            conn.commit()
            cursor.close()
            return "user_created"
        except Exception as e:
            print("add_user: "+str(e))
            return "failed_insert"
        
    def update_password(self,username,password):
        try:
            conn,cursor=self.connect_db()
            query = f"UPDATE user_data SET password = '{password}' WHERE username = '{username}'"
            cursor.execute(query)
            conn.commit()
            conn.close()
            return "password_updated"
        except Exception as e:
            print("update_password: "+str(e))
            return "update_failed"
        
    def pinecone_init(self):
        index_name = os.getenv('pinecone_index_name')

        # initialize connection to pinecone (get API key at app.pinecone.io)
        pinecone.init(
            api_key=os.getenv('pinecone_api_key'),
            environment=os.getenv('pinecone_environment') # find next to api key in console
        )
        # check if 'openai' index already exists (only create index if not)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=len(embeds[0]))
        # connect to index
        index = pinecone.Index(index_name)
        return index
    
    def query_pinecone(self,query,resturant_id):
        index=self.pinecone_init()
        openai.api_key = os.getenv('open_api_key')
        MODEL=os.getenv('Embedding_Model')
        xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
        res = index.query([xq], top_k=20, include_metadata=True,namespace=resturant_id)
        #return res['matches']
        reviews=[]
        for match in res['matches']:
            reviews.append(f"{match['score']:.2f}: {match['metadata']['text']}")
        return reviews
    
    def chat_gpt(self,query,prompt):
        openai.api_key = os.getenv('open_api_key')
        response_summary =  openai.ChatCompletion.create(
            model = "gpt-3.5-turbo", 
            messages = [
                {"role" : "user", "content" : f'{query} {prompt}'}
            ],
            temperature=0
        )
        return response_summary['choices'][0]['message']['content']