import openai
from openai.error import OpenAIError
import re 
import os
from dotenv.main import load_dotenv
import logging
import inspect
from .answers import actual_answers
from .Database_functions import MongoUpdateTotalMark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs.log"),
    ],)

 # take environment variables from .env.


class Evaluate:

    def __init__(self,subject,avilable_answers,user_id) -> None:
        self.subject = subject
        self.avilable_answers = avilable_answers
        self.indi_mark = [0 for i in range(15)]
        self.user_id = user_id
        logger = logging.getLogger("Evaluate")
        logger.info("Evaluate object created")

    def generate_chat_response(self,prompt):
        load_dotenv()
        openai.api_key = os.getenv('KEY')
        logger = logging.getLogger("generate_chat_response")
        try:
            # Create a completion request with the specified engine, prompt, and max tokens.
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1024
            )
            logger.info("GPT-3 response generated")
            return response.choices[0].message.content

        except OpenAIError as error:
            # Handle API errors.
            error_message = error.__class__.__name__ + ': ' + str(error)
            print('API Error:', error_message)
            logger.error(error_message)
            return None

        except Exception as e:
            # Handle other exceptions.
            print('Exception:', str(e))
            logger.error(str(e))
            return None


    def generate_prompt(self,questions):
        
        logger = logging.getLogger("generate_prompt")
        n=1
         
        prompt = """task 1 : I will give you many sets of pair of  sentences that must be compared as input, i want you to give me output only in the format I am describing. I dont want anything else as the output.". 

        task 2: I will give you 2 sets of pair of  sentences .I want you to take only the sentence 2 and give me all the tags(single letter concepts) that i am strong at inside the <strong></strong> tag and things i am weak at inside <weak></weak> tag, it must be array of words inside both the tags.i want you to give me output only in the format I am describing. I dont want anything else as the output. I will fill the space in the output where you must insert your output value as "{}".

        task 3: Give me an suggestion on what to learn and how to learn to improve my knowledge based on my answer in sentence 2. give it in <suggest></suggest> tag

        DO'S:
        1)when sentence 2 is not relevant to sentence 1 give rating as 0
        2)output only in the format i specify
        3)output range from 1 to 10
        4)if sentence 1 and sentence 2 are conceptually similar give output based on similarity
        DONT'S
        1)Any other text other than the specified output 
        2)No explination and reasoning
        3)no assurance texts like "Understood. Here's the output in the exact format you described, based on the input sentences you provided:"

        sample output  for each sets:
        <rating>number</rating>
        <strong>['strength1','stregth2']</strong>
        <weak>['weakness1','weakness2']</weak> *required*
        <suggest>suggestion</suggest>\n Input : \n"""



        for i in questions[self.subject]:
            if questions[self.subject][i] == "":
                continue
            else:
                q = questions[self.subject][i]
                a = actual_answers[self.subject][i]
                
                para1="\nset "+str(n)+"\n"+"\n[sentence 1:"+a+"\n"
                para2="sentence 2:" +q+"\n"
                prompt+=para1+para2+"]\n"
                n+=1
        
        logger.info("prompt for the AI generated")
        return prompt


    def extraction(self,input_text):
        logger = logging.getLogger("extraction")
        pattern = r'<rating>(.*?)<\/rating>'
        pattern2= r'<strong>(.*?)<\/strong>'
        pattern3= r'<weak>(.*?)<\/weak>'
        pattern4= r'<suggest>(.*?)<\/suggest>'
        scores=[]
        scores.append(re.findall(pattern, input_text))
        scores.append(re.findall(pattern2, input_text))
        scores.append(re.findall(pattern3, input_text))
        scores.append(re.findall(pattern4, input_text))

        logger.info("scores extracted from the AI generated response")
        return scores
    
    def calculate_percentage(self,scores):

        
        logger = logging.getLogger("calculate_percentage")
        for x in range(len(scores[0])):
            self.indi_mark[int(self.avilable_answers[x])-1] += int(scores[0][x]) 

        print("this iss the problem ------->",self.indi_mark)

        final_score = {}

        if "dbms" in self.subject:
            final_score = {
                "Relational Databases": 0,
                "Database Design": 0,
                "Transactions and Concurrency": 0,
                "Data Storage and Querying": 0,
                "Advanced topics": 0,
                "totalMarks": 0
            }

            for i in range(len(self.indi_mark)):
                if i <= 3:
                    final_score['Relational Databases'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 3 < i <= 6:
                    final_score['Database Design'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 6 < i <= 9:
                    final_score['Transactions and Concurrency'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 9 < i <= 12:
                    final_score['Data Storage and Querying'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 12 < i <= 15:
                    final_score['Advanced topics'] += int((int(self.indi_mark[i]) / 30) * 100)
                else:
                    continue

            final_score['totalMarks'] += (final_score['Relational Databases'] + final_score['Database Design'] + final_score['Transactions and Concurrency'] + final_score['Data Storage and Querying'] + final_score['Advanced topics'])/50

            

            if "EntryTest" in self.subject:
                MongoUpdateTotalMark(final_score,self.user_id,"entryTest","dbms","m2")
            else:
                MongoUpdateTotalMark(final_score,self.user_id,"exitTest","dbms","m2")

        elif "os" in self.subject:
            final_score = {
                "Operating System Overview": 0,
                "Process Management": 0,
                "Storage Management and File System": 0,
                "I/O Systems": 0,
                "Case Study": 0,
                "totalMarks": 0
            }

            for i in range(len(self.indi_mark)):
                if i <= 3:
                    final_score['Operating System Overview'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 3 < i <= 6:
                    final_score['Process Management'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 6 < i <= 9:
                    final_score['Storage Management and File System'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 9 < i <= 12:
                    final_score['I/O Systems'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 12 < i <= 15:
                    final_score['Case Study'] += int((int(self.indi_mark[i]) / 30) * 100)
                else:
                    continue

            final_score['totalMarks'] += (final_score['Operating System Overview'] + final_score['Process Management'] + final_score['Storage Management and File System'] + final_score['I/O Systems'] + final_score['Case Study'])/50

            if "EntryTest" in self.subject:
                MongoUpdateTotalMark(final_score,self.user_id,"entryTest","os","m2")
            else:
                MongoUpdateTotalMark(final_score,self.user_id,"exitTest","os","m2")

        elif "cn" in self.subject:
            final_score = {
                "Introduction and Physical layer": 0,
                "Data link layer and LAN": 0,
                "Network and Routing": 0,
                "Transport layer": 0,
                "Application layer": 0,
                "totalMarks": 0
            }

            for i in range(len(self.indi_mark)):
                if i <= 3:
                    final_score['Introduction and Physical layer'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 3 < i <= 6:
                    final_score['Data link layer and LAN'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 6 < i <= 9:
                    final_score['Network and Routing'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 9 < i <= 12:
                    final_score['Transport layer'] += int((int(self.indi_mark[i]) / 30) * 100)
                elif 12 < i <= 15:
                    final_score['Application layer'] += int((int(self.indi_mark[i]) / 30) * 100)
                else:
                    continue

            final_score['totalMarks'] += (final_score['Introduction and Physical layer'] + final_score['Data link layer and LAN'] + final_score['Network and Routing'] + final_score['Transport layer'] + final_score['Application layer'])/50

            if "EntryTest" in self.subject:
                MongoUpdateTotalMark(final_score,self.user_id,"entryTest","cn","m2")
            else:
                MongoUpdateTotalMark(final_score,self.user_id,"exitTest","cn","m2")

        logger.info("final score calculated")
        return final_score
        
    
    def mcqPercentage(self,score):
        logger = logging.getLogger("mcqPercentage")
        
        if 'c/c++' in self.subject:
            final_score = {
                "Basic Syntax and Language Fundamentals":0,
                "Functions and Scope":0,
                "Arrays and Pointers":0,
                "Object-Oriented Programming (C++)":0,
                "File Handling and Input/Output":0,
                "totalMarks":0
            }

            for i in range(1,len(score)+1):
                if i <= 3:
                    final_score['Basic Syntax and Language Fundamentals'] += int((int(score[str(i)]) / 3) * 100)
                elif 3 < i <= 6:
                    final_score['Functions and Scope'] += int((int(score[str(i)]) / 3) * 100)
                elif 6 <  i <= 9:
                    final_score['Arrays and Pointers'] += int((int(score[str(i)]) / 3) * 100)
                elif 9 < i <= 12:
                    final_score['Object-Oriented Programming (C++)'] += int((int(score[str(i)]) / 3) * 100)
                elif 12 < i <= 15:
                    final_score['File Handling and Input/Output'] += int((int(score[str(i)]) / 3) * 100)
                else:
                    continue

            final_score['totalMarks'] += (final_score['Basic Syntax and Language Fundamentals'] + final_score['Functions and Scope'] + final_score['Arrays and Pointers'] + final_score['Object-Oriented Programming (C++)'] + final_score['File Handling and Input/Output'])/50

            if "EntryTest" in self.subject:
                MongoUpdateTotalMark(final_score,self.user_id,"entryTest","c/c++","m1")
            else:
                MongoUpdateTotalMark(final_score,self.user_id,"exitTest","c/c++","m1")
            
            print(final_score)
            
        elif 'java' in self.subject:
            final_score = {
                "inheritance":0,
                "polymorphism":0,
                "encapsulation":0,
                "abstraction":0,
                "interfaces":0,
                "totalMarks":0
            }

            print(len(score))
            print(score)
            for i in range(1,len(score)+1):
                if i <= 3:
                    final_score['inheritance'] += int((int(score[str(i)]) / 3) * 100)
                elif 3 < i <= 6:
                    final_score['polymorphism'] += int((int(score[str(i)]) / 3) * 100)
                elif 6 < i <= 9:
                    final_score['encapsulation'] += int((int(score[str(i)]) / 3) * 100)
                elif 9 < i <= 12:
                    final_score['abstraction'] += int((int(score[str(i)]) / 3) * 100)
                elif 12 < i <= 15:
                    final_score['interfaces'] += int((int(score[str(i)]) / 3) * 100)
                else:
                    continue
        

            final_score["totalMarks"] = (final_score["inheritance"] + final_score["polymorphism"] + final_score["encapsulation"] + final_score["abstraction"] + final_score["interfaces"])/50

            if "EntryTest" in self.subject:
                MongoUpdateTotalMark(final_score,self.user_id,"entryTest","java","m1")
            else:
                MongoUpdateTotalMark(final_score,self.user_id,"exitTest","java","m1")
                
            print(final_score)
        
        elif "oops" in self.subject:
            final_score = {
                "classes": 0,
                "objects": 0,
                "constructors": 0,
                "methods": 0,
                "inheritance": 0,
                "totalMarks": 0
            }

            for i in range(1,len(score)+1):
                if i <= 3:
                    final_score['classes'] += int((int(score[str(i)]) / 3) * 100)
                elif 3 < i <= 6:
                    final_score['objects'] += int((int(score[str(i)]) / 3) * 100)
                elif 6 < i <= 9:
                    final_score['constructors'] += int((int(score[str(i)]) / 3) * 100)
                elif 9 < i <= 12:
                    final_score['methods'] += int((int(score[str(i)]) / 3) * 100)
                elif 12 < i <= 15:
                    final_score['inheritance'] += int((int(score[str(i)]) / 3) * 100)
                else:
                    continue

            final_score['totalMarks'] += (final_score['classes'] + final_score['objects'] + final_score['constructors'] + final_score['methods'] + final_score['inheritance'])/50

            if "EntryTest" in self.subject:
                MongoUpdateTotalMark(final_score,self.user_id,"entryTest","oops","m1")
            else:
                MongoUpdateTotalMark(final_score,self.user_id,"exitTest","oops","m1")

        elif "dsa" in self.subject:
            final_score = {
                "arrays": 0,
                "linkedLists": 0,
                "stacks": 0,
                "queues": 0,
                "trees": 0,
                "totalMarks": 0
            }

            for i in range(1,len(score)+1 ):
                if i <= 3:
                    final_score['arrays'] += int((int(score[str(i)]) / 3) * 100)
                elif 3 < i <= 6:
                    final_score['linkedLists'] += int((int(score[str(i)]) / 3) * 100)
                elif 6 < i <= 9:
                    final_score['stacks'] += int((int(score[str(i)]) / 3) * 100)
                elif 9 < i <= 12:
                    final_score['queues'] += int((int(score[str(i)]) / 3) * 100)
                elif 12 < i <= 15:
                    final_score['trees'] += int((int(score[str(i)]) / 3) * 100)
                else:
                    continue

            final_score['totalMarks'] += (final_score['arrays'] + final_score['linkedLists'] + final_score['stacks'] + final_score['queues'] + final_score['trees'])/50

            if "EntryTest" in self.subject:
                MongoUpdateTotalMark(final_score,self.user_id,"entryTest","dsa","m1")
            else:
                MongoUpdateTotalMark(final_score,self.user_id,"exitTest","dsa","m1")
            
            logger.info("mcq output generated")
        

        
    def jsonify(self,scores):

        logger = logging.getLogger("jsonify")
        output={}
        
        for i in scores:
            output[self.subject]={}
            for j in range (1,len(i)+1):
                output[self.subject][str(j)]={}
        a=0
        for subscore in zip(scores[0],scores[1],scores[2],scores[3]):
            print("this is subscoree ------>",subscore)
            suggest={"rating":subscore[0],
                     "Strength":subscore[1],
                     "Weak":subscore[2],
                     "Suggestion":subscore[3]}
            output[self.subject][str(self.avilable_answers[a])]=suggest
            a+=1
        
        final_score = self.calculate_percentage(scores)
        output['final_score'] = final_score

        logger.info("jsonified output generated")
        return output
  


