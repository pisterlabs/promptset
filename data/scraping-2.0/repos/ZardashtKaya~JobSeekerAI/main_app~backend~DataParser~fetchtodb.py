import sqlite3
con = sqlite3.connect("data.db",check_same_thread=False)
import re
cur = con.cursor()
import openai
# import openapi
# openapi.__init__()
user_skills=[]
user_info=[]

class Fetcher:
    # def __init__(self, *args,) -> None:
    #     pass
    @staticmethod
    def add_skill(skills=list):
        # make a temporary list of skills
        temp = []
        for skill in skills:
            skill = skill.strip()
            temp.append(skill)
            user_skills.append(skill)
        # convert to string while keeping commas
        temp = ','.join(temp)

        # make a string of skills in the database in one line

        skills_in_db = ''
        for skill in Fetcher.get_skills():
            skills_in_db += skill[1] + ','
       

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """ you are a system that compares two lists and only ouputs the needed data in a comma seperated format and nothing else it is not case sensitive, you need to make sure that your only output is in the following format: skill1,skill2,skill3 meaning that they are comma seperated and no spaces between them, avoid using filler words like 'sure here it is' i want you to only show me the result in a comma seperated format and nothing else, if there is existing skills, dont output them, only output the needed skills to add, and if there are no skills to add then output 'nothing to add' avoid using extra words and keep it simple, your output will be parsed and needs to be in the correct format that i tell you"""},
                
                {"role": "user", "content": """check if the folllowing skills """+temp+""" are in """ +skills_in_db+""+""" if not, output only the skills that are not there in the following format: skill1,skill2,skill3 meaning that they are comma seperated and no spaces between them, avoid using filler words like 'sure here it is' i want you to only show me the result in a comma seperated format and nothing else"""}
            ]
        )
        response = completion.choices[0].message.content.strip()
        completion2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Clean up the following list to only include skill names or job title names"""},
                
                {"role": "user", "content":response}
            ]
        )
        response2 = completion2.choices[0].message.content.strip()
        # check if "nothing" is in the response
        if "nothing" in response2:
            return
        elif "," in response2:
            # remove trailing dot if there is one
            if response2[-1] == '.':
                response2 = response2[:-1]
            
            
            skills_to_add = response2.split(',')
            for skill in skills_to_add:
                # remove trailing and leading white space if exists
                skill = skill.strip()
                # check if data is more than 3 words
                if len(skill.split(' ')) <= 4:
                    if "sorry" or "Sorry" not in skill:
                        # add skill to database
                        cur.execute("INSERT INTO skill (skill_name) VALUES (?)", (skill,))
                        con.commit()

    def get_skills():
        cur.execute('select * from skill')
        return cur.fetchall()
    
    def get_rating(emp_skill_id):
        cur.execute("SELECT rating FROM employee_skill WHERE employee_skill_id = ?", (emp_skill_id,))
        return cur.fetchone()
    
    @staticmethod
    def get_top_skills_by_rating(emp_skill_id):
        # sort top skills by percentage where employee_skill_id in rating table
        cur.execute("SELECT * FROM rating WHERE employee_skill_id = ? ORDER BY rating_percentage DESC", (emp_skill_id,))
        
        
        return cur.fetchall()
    
    @staticmethod
    def get_employee(emp_id):
        cur.execute("SELECT * FROM employee WHERE employee_id = ?", (emp_id,))
        return cur.fetchone()
    
    @staticmethod
    def get_employee_id(skill_id):
        con.commit()
        cur.execute("SELECT employee_id FROM employee_skill WHERE skill_id = ?", (skill_id,))
        emp_id=cur.fetchone()
        con.commit()
        cur.execute("SELECT employee_skill_id FROM employee_skill WHERE skill_id = ?", (skill_id,))
        emp_skill_id=cur.fetchone()
        return [emp_id,emp_skill_id]
   
        
    @staticmethod
    def get_skill(skillname):

        # use LIKE
        cur.execute("SELECT * FROM skill WHERE skill_name LIKE ?", ('%'+skillname+'%',))
        return cur.fetchone()
    
    def add_employee(info):
        info = info.split(',')
        info = [i.strip() for i in info]
        user_info.append(info)
        cur.execute("INSERT INTO employee (name, email, phone, address) VALUES (?,?,?,?)", (info[0],info[1],info[2],info[3]))
        con.commit()

    def add_rest(parsedCV,skill_list,employee_info):
        # search for skill id using LIKE in SQLite for approximate matches
        info = [i.strip() for i in employee_info]
        # user_info.append(info)
        # get employee id
        # check length of info
        # check if name exists in table
        cur.execute("SELECT name FROM employee WHERE name = ?", (info[0],))
        name = cur.fetchone()
        if not name:

            if len(info) == 4:
                cur.execute("INSERT INTO employee (name, email, phone, address) VALUES (?,?,?,?)", (info[0],info[1],info[2],info[3]))
            elif len(info) ==3:
                cur.execute("INSERT INTO employee (name, email, phone, address) VALUES (?,?,?,?)", (info[0],info[1],info[2],"N/A"))
            elif len(info) ==2:
                cur.execute("INSERT INTO employee (name) VALUES (?)", (info[0],))
            
            con.commit()
            for skill in skill_list:
                cur.execute("SELECT skill_id FROM skill WHERE skill_name LIKE ?", ('%'+skill+'%',))
                skill_id = cur.fetchone()
                if skill_id:
                    

                    cur.execute("SELECT employee_id FROM employee WHERE name = ?", (info[0],))
                    id = cur.fetchone()
                    # convert to integer
                    id = id[0]
                    # insert into employee
                    cur.execute("INSERT INTO employee_skill (employee_id, skill_id) VALUES (?,?)", (id,skill_id[0]))
                    con.commit()
                    #insert into rating (employee_skill_id, rating_percentage) get employee_skill_id from inserted data in employee_lookup using employee_id and skill_id
                    # get employee_skill_id
                    cur.execute("SELECT employee_skill_id FROM employee_skill WHERE employee_id = ? AND skill_id = ?", (id,skill_id[0]))

                    emp_skill_id = cur.fetchone()[0]
                    # get rating_percentage by looking up current skill name in parsedCV
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": """what is the rating for the specific skill()"""+skill+""") in the following data """+parsedCV},
                            
                            {"role": "user", "content": """rating is:("""}
                        ]
                    )
                    rating_percentage = completion.choices[0].message.content.strip()
                    if ":" in rating_percentage:
                        rating_percentage = rating_percentage.split(':')[1]
                    elif "rating is" in rating_percentage:
                            rating_percentage = rating_percentage.split('rating is')[1]
                    elif "rating" in rating_percentage:
                        rating_percentage = rating_percentage.split('rating')[1]
                    elif "for" in rating_percentage:
                        rating_percentage = rating_percentage.split('for')[0]
                    rating_percentage.strip()
                    # regex to remove all non numeric characters except for .
                    rating_percentage = re.sub("[^0-9.]", "", rating_percentage)
                    # remove last character if it is a .
                    if rating_percentage[-1] == '.':
                        rating_percentage = rating_percentage[:-1]
                    

                    # convert to float
                    rating_percentage = float(rating_percentage)

                    cur.execute("INSERT INTO rating (employee_skill_id, rating_percentage) VALUES (?,?)", (emp_skill_id,rating_percentage))
                    con.commit()
                

