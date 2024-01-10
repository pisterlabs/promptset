import mysql.connector
import pymysql.cursors
import re
from mysql.connector import errorcode
import os
import openai

# insert new user into database
def DataUpdate(name, Email, dispatcher):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")
    mycursor = mydb.cursor()
    sql = "INSERT INTO rasa_person (name, Email) VALUES (%s, %s)"
    vals = (name, Email)
    mycursor.execute(sql, vals)
    mydb.commit()


def dataQuery(person_id, dispatcher):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")

    mycursor = mydb.cursor()
    sql = "SELECT name from rasa_person where id='%s'" % (person_id)

    try:
        # Execute the SQL Query
        mycursor.execute(sql)
        results = mycursor.fetchall()
        for row in results:
            name = row[0]
            return name
    except:
        dispatcher.utter_message("Error : Unable to fetch data.")


def dataGetId(name, Email, dispatcher):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")

    mycursor = mydb.cursor()
    sql = "SELECT id FROM rasa_person WHERE name='%s' AND Email='%s'" % (name, Email)

    try:
        # Execute the SQL Query
        mycursor.execute(sql)
        results = mycursor.fetchall()
        for row in results:
            id = row[0]
            # Now print fethced data
            dispatcher.utter_message(f"Your ID is {id}, remember this so we can pick up where we left off next time!")
            return id

    except:
        dispatcher.utter_message("Error : Unable to fetch data.")


# For a returning user so that last topic and correct Qs can be displayed
def dataGetPrevQ(id, dispatcher):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")

    mycursor = mydb.cursor()
    sql = "SELECT level_sql,last_question_score,name FROM rasa_person WHERE id='%s'" % (
        id)  # select level and score from user table

    try:
        # Execute the SQL Query
        mycursor.execute(sql)
        results = mycursor.fetchall()

        for row in results:
            level_sql = row[0]
            last_question_score = row[1]
            name = row[2]
            sql2 = "SELECT q_name,Question_id FROM SQL_level WHERE level='%s'" % (
                level_sql)  # get question topic and id from SQL_level table

            mycursor.execute(sql2)
            results2 = mycursor.fetchall()

            for rows in results2:
                q_name = rows[0]
                qid = rows[1]
                return q_name, level_sql, last_question_score, name, qid  # return to slots

    except:
        dispatcher.utter_message("Error : Unable to fetch data.")


# function to get the next question
def dataGetNewQ(name, id, dispatcher, level_sql, exercise_id, n_correct_qs):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")

    mycursor = mydb.cursor()
    sql = 'SELECT level_sql FROM rasa_person WHERE id="{0}";'.format(id)
    # check if they want same or next level
    if name == 'similar':
        # user wants to try a Q on the same topic
        exercise_id = int(exercise_id) + 1  # next question, same topic
        sql2 = "SELECT question,Question_id FROM SQL_level WHERE level='%s' AND Question_id ='%s'" % (
            level_sql, exercise_id)
        try:
            # Execute the SQL Query
            mycursor.execute(sql2)
            results = mycursor.fetchall()
            for row in results:
                question = row[0]
                q_id = row[1]
                return question, q_id, level_sql, n_correct_qs
        except:
            dispatcher.utter_message("Error : Unable to fetch data.")

    if name == 'move_on':
        # user wants to move onto a harder topic
        n = 0  # reset number of correct Qs per topic
        level_sql = level_sql + 1  # next topic
        sql2 = "SELECT question,Question_id FROM SQL_level WHERE level='%s'" % (level_sql)
        try:
            # Execute the SQL Query
            mycursor.execute(sql2)
            results = mycursor.fetchall()
            for row in results:
                question = row[0]
                q_id = row[1]
                return question, q_id, level_sql, n

        except:
            dispatcher.utter_message("Error : Unable to fetch data.")


# function to check the users answer
def dataCheckAnswer(answer, exercise_id, dispatcher, id, n_correct_qs):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")

    mycursor = mydb.cursor()
    sql = "SELECT answer FROM SQL_level WHERE Question_id='%s'" % (exercise_id)
    sql_name = 'SELECT name from rasa_person where id="{0}";'.format(id)
    try:
        # Execute the SQL Query
        mycursor.execute(sql)
        results = mycursor.fetchall()

        for row in results:
            ans = row[0]
            if answer == ans:
                # Correct: when users answer equals correct answer
                dispatcher.utter_message(f"Correct! Well done\n")
                dispatcher.utter_message(f"Answer: {answer}")  # answer displayed
                reply2 = "Correct"  # reply set to "Correct"
                n = n_correct_qs + 1  # increase no. correct questions on topic by 1
                return reply2, n

            else:
                sql2 = answer  # set user answer to a new sql query
                try:
                    # Execute the SQL Query
                    mycursor.execute(sql2)  # execute users answer
                except mysql.connector.Error as err:
                    # users answer throws a database error
                    dispatcher.utter_message("Hmm looks like something went wrong, heres some feedback:\n")
                    dispatcher.utter_message(f"Message: {err.msg} \n")  # error message given as feedback
                    dispatcher.utter_message(response="utter_show_answer")  # reminds user they can skip/see answer
                    reply2 = "Incorrect"  # set reply to "Incorrect"
                    return reply2, n_correct_qs

                # if users answer does not throw error, compare results sets
                results2 = mycursor.fetchall()  # fetch users results set
                mycursor.execute(ans)  # execute correct answer to get results set
                results3 = mycursor.fetchall()  # fetch correct results set
                difference = 0
                if len(results3) > 1:
                    # if more than one result in the results set
                    for x, y in zip(results3, results2):
                        new_list_diff = list(
                            set(x).difference(y))  # get set of both results sets (equal values) and get difference
                        if len(new_list_diff) > 0:  # if theres a difference in the 2 results sets then enter
                            difference = difference + 1
                    if difference == 0:
                        # if there results sets are the same
                        dispatcher.utter_message(
                            f"Looks like this is correct! However I got a slightly different answer: {ans}")  # let user know its not exactly what was wanted
                        reply2 = "Correct"  # set reply to "Correct"
                        n = n_correct_qs + 1  # increase no. correct answers by 1
                        return reply2, n
                    if difference > 0:
                        # if the results sets are different
                        dispatcher.utter_message(f"Not quite... Try again!")
                        reply2 = "Incorrect"  # set reply to "Incorrect"
                        return reply2, n_correct_qs

                if len(results3) == 1:
                    # if theres only one results in results set
                    if results2 == results3:
                        dispatcher.utter_message(
                            f"Looks like this is correct! However I got a slightly different answer: {ans}")
                        reply2 = "Correct"
                        n = n_correct_qs + 1
                        return reply2, n
                    else:
                        # if the results sets are different
                        dispatcher.utter_message(f"Not quite, the results set should be:{results3}. Try again!")
                        reply2 = "Incorrect"  # set reply to "Incorrect"
                        return reply2, n_correct_qs


    except:
        dispatcher.utter_message("Error : Unable to fetch data.")


# function to update database and give lesson summary when user wants to quit
def dataUpdateOnQuit(id, level, dispatcher, name, last_question_score):
    global level_s
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")
    mycursor = mydb.cursor()
    # get starting level
    sql2 = "SELECT level_sql FROM rasa_person WHERE id='%s'" % (id)  # get users level at start of lesson
    try:
        # Execute the SQL Query
        mycursor.execute(sql2)
        results2 = mycursor.fetchall()
        for row in results2:
            level_s = row[0]
        dispatcher.utter_message("------ LESSON SUMMARY --------\n")
        dispatcher.utter_message("Types of questions attempted:\n")
        for i in range(level_s, level + 1):  # iterate from level at start of lesson to current level (from slot)
            sql3 = "SELECT DISTINCT q_name FROM SQL_level WHERE level='%s' LIMIT 1" % (
                i)  # get topic name so it can be displayed for user
            try:
                # Execute the SQL Query
                mycursor.execute(sql3)
                results3 = mycursor.fetchall()
                for rows in results3:
                    q_name = rows[0]
                    dispatcher.utter_message(f">> {q_name}")  # display topic name
            except:
                dispatcher.utter_message("Error : Unable to fetch data.1")

        dispatcher.utter_message("-------------------------\n")
    except:
        dispatcher.utter_message("Error : Unable to fetch data. 2")

    sql = "UPDATE rasa_person SET level_sql='%s',last_question_score='%s' WHERE id='%s'" % (
        level, last_question_score, id)  # update user table to reflect the lesson
    mycursor.execute(sql)
    mydb.commit()
    dispatcher.utter_message(f" Bye {name}! I hope to see you again soon")


# function to answer a users general question
def dataGeneralQuestion(message, level, dispatcher):
    msg = message.lower()  # put most recent user input in lower case
    if 'syntax' in msg:  # if 'syntax' is in question then answer with syntax only information
        # check what level (topic) the user is on so that the question can be answered in context of question they are on
        if level == 1:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT statement? \n Try this syntax in your exercise: \n SELECT column1, column2  \n FROM table_name;")
        if level == 2:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT statement? \n Try this syntax in your exercise: \n SELECT * FROM table_name;")
        if level == 3:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT DISTINCT statement? \n Try this syntax in your exercise: \n SELECT DISTINCT column1, column2 \n FROM table_name;")
        if level == 4:
            dispatcher.utter_message(
                f" Remember you learned about using WHERE in a SELECT statement when conditions are involved? \n Try this syntax in your exercise: \n SELECT column1, column2 \n FROM table1 WHERE condition")
        if level == 5:
            dispatcher.utter_message(
                f" Remember you learned about WHERE with AND/OR/NOT in a SELECT statement when multiple conditions are involved? \n Try this syntax in your exercise: \n SELECT column1, column2 \n FROM table1 WHERE condition1 AND condition2")
        if level == 6:
            dispatcher.utter_message(
                f" Remember you learned about WHERE with AND/OR/NOT in a SELECT statement when multiple conditions are involved? \n Try this syntax in your exercise: \n SELECT column1, column2 \n FROM table1 WHERE condition1 AND (condition2 OR condition3)")

    elif 'theory' in msg: # if 'theory' is in question then answer with theory only information
        # check what level (topic) the user is on so that the question can be answered in context of question they are on
        if level == 1:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT statement?  \n SELECT - extracts data from a database.")
        if level == 2:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT * statement?  \n SELECT * - returns all values in a table.")
        if level == 3:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT DISTINCT statement? \n The SELECT DISTINCT statement is used to return only distinct (different) values. \n Inside a table, a column often contains many duplicate values and sometimes you only want to list the different (distinct) values.")
        if level == 4:
            dispatcher.utter_message(
                f" Remember you learned about using WHERE in a SELECT statement when conditions are involved? \n Including WHERE in a SELECT statement allows us to include conditions in the query. \n These conditions can include using =, <, > ")
        if level == 5:
            dispatcher.utter_message(
                f" Remember you learned about WHERE with AND/OR/NOT in a SELECT statement when multiple conditions are involved? \n We use WHERE with AND when there are multiple conditions involved so that all conditions \n can be accounted for.")
        if level == 6:
            dispatcher.utter_message(
                f" Remember you learned about WHERE with AND/OR/NOT in a SELECT statement when multiple conditions are involved? \n In this question more than one type of operator is used. \n This means there are more than 2 conditions which use a combination of operators. \n For example AND with OR..")

    else:   # if the question is general then give an explanation of the topic they are on

        if level == 1:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT statement? \n Try this syntax in your exercise: \n SELECT column1, column2  \n FROM table_name;")
        if level == 2:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT statement? \n Try this syntax in your exercise: \n SELECT * FROM table_name;")
        if level == 3:
            dispatcher.utter_message(
                f" Remember you learned about the SELECT DISTINCT statement? \n Try this syntax in your exercise: \n SELECT DISTINCT column1, column2 \n FROM table_name;")
        if level == 4:
            dispatcher.utter_message(
                f" Remember you learned about using WHERE in a SELECT statement when conditions are involved? \n Try this syntax in your exercise: \n SELECT column1, column2 \n FROM table1 WHERE condition")
        if level == 5:
            dispatcher.utter_message(
                f" Remember you learned about WHERE with AND/OR/NOT in a SELECT statement when multiple conditions are involved? \n Try this syntax in your exercise: \n SELECT column1, column2 \n FROM table1 WHERE condition1 AND condition2")
        if level == 6:
            dispatcher.utter_message(
                f" Remember you learned about WHERE with AND/OR/NOT in a SELECT statement when multiple conditions are involved? \n Try this syntax in your exercise: \n SELECT column1, column2 \n FROM table1 WHERE condition1 AND (condition2 OR condition3)")


# Give a first time user a question
def FirstTimeQustion(dispatcher):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")
    mycursor = mydb.cursor()
    sql2 = "SELECT question,Question_id, q_name FROM SQL_level WHERE level='1' AND Question_id ='1'"    # just give them first question
    try:
        # Execute the SQL Query
        mycursor.execute(sql2)
        results = mycursor.fetchall()

        for row in results:
            question = row[0]
            q_id = row[1]
            qname = row[2]
            return question, q_id, qname
    except:
        dispatcher.utter_message("Error : Unable to fetch data.")


# a user skips a question or requests to see the answer
def dataShowAnswer(dispatcher, e_id):
    mydb = mysql.connector.connect(host="localhost", user="root",
                                   passwd="****", database="Rasa_feedback")

    mycursor = mydb.cursor()
    # mycursor_name = mydb.cursor()
    sql = "SELECT answer,question FROM SQL_level WHERE Question_id='%s'" % (e_id)   # get the answer and the question
    try:
        # Execute the SQL Query
        mycursor.execute(sql)
        results = mycursor.fetchall()

        for row in results:
            answer = row[0]
            question = row[1]

            dispatcher.utter_message(f"The answer to: '  {question}  ' is: \n ")    # remind user of the question
            dispatcher.utter_message(f" {answer} ") # display answer for user

    except:
        dispatcher.utter_message("Error : Unable to fetch data.")


