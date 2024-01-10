import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
prompt="#create a flask route and method to check if a given string contains digits\n"

def get_flask_response(prompt):
    response = openai.Completion.create(
    engine="davinci-codex",
    prompt=prompt,
    temperature=0,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["#"]
    )
    resp=response.choices[0].text
    return resp
#print(get_flask_response(prompt))


"""
#Write SQL code to left inner join Employee table with Manager Table.
Query=

#Write SQL code to find Students with secondName ending with letters "th".
sql query="SELECT * FROM Students WHERE secondName LIKE '%th'"

#Write SQL code to find EmployeeID of managers who handle more than 5 employess.
sql query="SELECT EmployeeID FROM Employee WHERE EmployeeID IN (SELECT ManagerID FROM Employee GROUP BY ManagerID HAVING COUNT(*) > 5)"

#Write SQL code to find the number of students who have enrolled into more than 4 classes.
sql query="SELECT COUNT(*) FROM student_courseenrollment WHERE course_id IN (SELECT course_id FROM student_courseenrollment GROUP BY course_id HAVING COUNT(*) > 4)"


#Write SQL code to create a stored procedure that takes employeeID as input and returns salary in Employee Table.
sql query=


"""
def get_SQL_response(prompt):
    response = openai.Completion.create(
    engine="davinci-codex",
    prompt=prompt+"\nsql query=",
    temperature=0.1,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["#", ";"]
    )
    resp=response.choices[0].text
    return resp


def get_code_explaination(prompt):
    response = openai.Completion.create(
    engine="davinci-codex",
    prompt=prompt+"\"\"\"\nHere's what the above Code is doing:\n",
    temperature=0,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\"\"\""]
    )
    resp=response.choices[0].text
    return resp