#!/usr/bin/python coding=utf-8
# !/bin/bash

"""
// ---------------------------------------------------------------------------
//
//	Security by Design for Cloud, Mobile and IoT Ecosystem (SecD4CLOUDMOBILE)
//
//  Copyright (C) 2020 Instituto de Telecomunicações (www.it.pt)
//  Copyright (C) 2020 Universidade da Beira Interior (www.ubi.pt)
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// This Work was developed under Doctoral Grant, supported by project 
// CENTRO-01-0145-FEDER-000019 - C4 - Competence Center in Cloud Computing, 
// Research Line 1: Cloud Systems, Work package WP 1.2 - Systems design and development 
// processes and software for Cloud and Internet of Things ecosystems, cofinanced by the 
// European Regional Development Fund (ERDF) through the Programa Operacional Regional do 
// Centro (Centro 2020), in the scope of the Sistema de Apoio à Investigação Científica e 
// Tecnológica - Programas Integrados de IC&DT.
// ---------------------------------------------------------------------------
"""

import os
from sys import exit
import webbrowser
from markdown import markdown
from xhtml2pdf import pisa
from switch import Switch
from PyPDF2 import PdfFileMerger
import openai

################################# GLOBAL VARIABLES #################################

openai.api_key="Your API Keys"

version = 1

# list that contains to answers in the written file
input_list = []

# add the answers (output) to a list to write as the respective answers and comments in the generated file with answers
answers_list = []
comments_list = []
table_for_report = []

# create a dictionairy to store the answers to the questions
# question_and_answers as questions_and_answers
questions_and_answers = {
    "Q1": "",
    "Q2": "",
    "Q3": "",
    "Q4": "",
    "Q5": "",
    "Q6": "",
    "Q7": "",
    "Q8": "",
    "Q9": "",
    "Q10": "",
    "Q11": "",
    "Q12": "",
    "Q13": "",
    "Q14": "",
    "Q15": "",
    "Q16": "",
    "Q17": "",
    "Q18": "",
    "Q19": "",
    "Q20": "",
    "Q21": "",
    "Q22": ""
}

# Questions
# Q1   -> Mobile Platform
# Q2   -> Application Domain Type
# Q3   -> Authentication 
# Q4   -> Authentication Schemes
# Q5   -> Has a DB
# Q6   -> Type of data storage
# Q7   -> DBMS name
# Q8   -> Type of data
# Q9   -> Local of data storage
# Q10  -> User Registration
# Q11  -> Way of user registration
# Q12  -> Programming Languages
# Q13  -> Input Forms
# Q14  -> Upload Files
# Q15  -> Logs
# Q16  -> System Regular Updates
# Q17  -> Third-party Software
# Q18  -> System Cloud Environment
# Q19  -> Hardware Specifications
# Q20  -> Hardware Auth
# Q21  -> Hardware Communications
# Q22  -> Data Center Physical Access


# TO -DO -> in case of answer "others" (user input),
# at the time of execution add to respective dict

question1 = {
    "1": "Android App",
    "2": "iOS App",
    "3": "Web Application",
    "4": "Hybrid Application",
    "5": "Harmony OS App",
    "6": "Tizen Application",
    "7": "IoT System",
    "8": "Chrome OS Application",
    "9": "Ubuntu Touch Application",
    "10": ""
}

question2 = {
    "1": "Entertainment",
    "2": "Mobile Game",
    "3": "m-Commerce ",
    "4": "m-Health",
    "5": "m-Learning",
    "6": "m-Payment",
    "7": "m-Social Networking",
    "8": "Multi User Collaboration",
    "9": "m-Tourism",
    "10": "Smart Agriculture",
    "11": "Smart Air Quality",
    "12": "Smart Healthcare",
    "13": "Smart Home",
    "14": "Smart Manufacturing",
    "15": "Smart Transportation",
    "16": "Smart Waste Monitoring",
    "17": "Smart Wearables",
    "18": "Smart Home"
}

question3 = {
    "1": "Yes",
    "2": "No"
}

question4 = {
    "1": "Biometric-based authentication",
    "2": "Channel-based authentication",
    "3": "Factors-based authentication",
    "4": "ID-based authentication",
    "5": ""
}

question5 = {
    "1": "Yes",
    "2": "No"
}

question6 = {
    "1": "SQL (Relational Database)",
    "2": "NoSQL"
}

question7 = {
    "1": "SQL Server",
    "2": "MySQL",
    "3": "PostgreSQL",
    "4": "SQLite",
    "5": "OracleDB",
    "6": "MariaDB",
    "7": "Cassandra",
    "8": "CosmosDB",
    "9": "MongoDB",
    "10": "JADE",
    "11": "HBase",
    "12": "Neo4J",
    "13": "Redis",
    "14": ""
}

question8 = {
    "1": "Personal Information",
    "2": "Confidential Data",
    "3": "Critical Data",
    "4": ""
}

question9 = {
    "1": "Local Storage (Centralized Database)",
    "2": "Remote Storage (Cloud Database)",
    "3": "Both"
}

question10 = {
    "1": "Yes",
    "2": "No"
}

question11 = {
    "1": "The users will register themselves",
    "2": "Will be an administrator that will register the users"
}

question12 = {
    "1": "C#",
    "2": "C/C++/Objective-C",
    "3": "HTML5 + CSS + JavaScript",
    "4": "Java",
    "5": "Javascript",
    "6": "PHP",
    "7": "Python",
    "8": "Ruby",
    "9": "Kotlin",
    "10": "Swift",
    "11": ""
}

question13 = {
    "1": "Yes",
    "2": "No"
}

question14 = {
    "1": "Yes",
    "2": "No"
}

question15 = {
    "1": "Yes",
    "2": "No"
}

question16 = {
    "1": "Yes",
    "2": "No"
}

question17 = {
    "1": "Yes",
    "2": "No"
}

question18 = {
    "1": "Community Cloud",
    "2": "Hybrid Cloud",
    "3": "Private Cloud",
    "4": "Public Cloud",
    "5": "Virtual Private Cloud"
}

question19 = {
    "1": "Yes",
    "2": "No"
}

question20 = {
    "1": "No Authentication",
    "2": "Symmetric Key",
    "3": "Basic Authentication (user/pass)",
    "4": "Certificates (X.509) ",
    "5": "TPM (Trusted Platform Module)"
}

question21 = {
    "1": "3G",
    "2": "4G/LTE",
    "3": "5G",
    "4": "GSM (2G)",
    "5": "Bluetooth ",
    "6": "Wi-Fi ",
    "7": "GPS ",
    "8": "RFID ",
    "9": "NFC",
    "10": "ZigBee",
    "11": ""
}

question22 = {
    "1": "Yes",
    "2": "No"
}

'''
>question template
def NAME():
    print("  : ")
    print("")
    print( "  1 -   ")
    print( "  2 -   ")
    print( "  3 -   ")
    print( "  4 -   ")
    print("")
'''

################################# FUNCTIONS #################################

"""
 [Summary]: Common method that validates the filename entered by the user and checks if the file exists or not
 [Returns]: No return
"""
def readInputFromFile():
    # user inputs the file name and checks if the file exists
    while True:
        filepath = validateInput(2)
        if not os.path.isfile(filepath):
            print("File path {} does not exist...".format(filepath))
        else:
            break

    with open(filepath, 'r') as file:
        line = file.readline()
        while line:
            # print (line.strip())
            # print (line.split('#')[0].strip() )

            # read line until character '#' which means after that is a comment
            input_list.append(line.split('#')[0].strip())
            line = file.readline()


"""
[Summary]: Common method to validate input and implemts dynamic arguments
[Arguments]: 
    - arg(1) what to validate -> if it is to validate a int or a string (1 or 2)
    - arg(2) n_options -> number of options in the question (==range)
[Returns]: Returns user inputs
"""
def validateInput(*arg):

    while True:

        # validate a int
        if arg[0] == 1:
            try:
                user_input = input("  > ")
                user_input = int(user_input)

            # syntax error, name error
            except (SyntaxError, NameError, TypeError):
                print("  Not a valid answer!  ")
                print("")
                continue
            else:
                if (type(user_input) is int) and (user_input in range(0, arg[1])):
                    break
                else:
                    print("  Not a valid answer!  ")
                    print("")

        # validate a string
        if arg[0] == 2:
            try:
                user_input = input("  > ")

            # syntax error, name error
            except (SyntaxError, NameError, TypeError):
                print("  Not a valid answer!  ")
                print("")
                continue
            else:
                if type(user_input) is str:
                    break
                else:
                    print("  Not a valid answer!  ")
                    print("")

    return user_input


"""
[Summary]: Method that gets the platform of the mobile application developed or to be developed
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def arqui(version):
    print("---")
    print("")
    if version == 1:
        print("  **Which will be the mobile platform of the system?**  ")
    else:
        print("  **What is the mobile platform of the system?**  ")
    print("  (This is a multiple choice question. Enter several options and end with 0.)  ")
    print("")
    print("  1 - Android Application  ")
    print("  2 - iOS Application  ")
    print("  3 - Web Application  ")
    print("  4 - Hybrid Application  ")
    print("  5 - Harmony OS App  ")
    print("  6 - Tizen Application  ")
    print("  7 - IoT System  ")
    print("  8 - Chrome OS Application  ")
    print("  9 - Ubuntu Touch Application  ")
    print("  10 - Others  ")
    print("")

    # function input() interprets the input
    # get user input differs from python 2.x and 3.x --> input() = version 3 | raw_input() = version 2.x

    while (1):
        # validate a integer (arg[0]==1 and specify the number available options(arg[1]==10))
        value = validateInput(1, 11)
        if value == 0:
            return
        if value == 10:
            print("  Please specify the mobile platform: (name between single quotes)  ")
            value2 = validateInput(2)
            questions_and_answers["Q1"] = questions_and_answers["Q1"] + str(value2) + ";"

        else:
            questions_and_answers["Q1"] = questions_and_answers["Q1"] + str(value) + ";"

"""
[Summary]: Method that gets the domain of the mobile application developed or to be developed
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def domain(version):
    print("---")
    print("")
    if version == 1:
        print("  **Which will be the domain of the system ?**  ")
    else:
        print("  **What is the domain of the system ?**  ")

    print("  (This is a multiple choice question. Enter several options and end with 0.)  ")
    print("")
    print("  1 - Entertainment ")
    print("  2 - Mobile Game ")
    print("  3 - m-Commerce ")
    print("  4 - m-Health ")
    print("  5 - m-Learning ")
    print("  6 - m-Payment ")
    print("  7 - m-Social Networking ")
    print("  8 - Mult User Collaboration ")
    print("  9 - m-Tourism ")
    print(" 10 - Smart Agriculture ")
    print(" 11 - Smart Air Quality ")
    print(" 12 - Smart Healthcare ")
    print(" 13 - Smart Home ")
    print(" 14 - Smart Manufacturing ")
    print(" 15 - Smart Transportation ")
    print(" 16 - Smart Waste Monitoring ")
    print(" 17 - Smart Wearables ")
    print(" 18 - Smart City ")
    print("")

    value = validateInput(1, 19)
    questions_and_answers["Q2"] = str(value)

"""
[Summary]: Method that allows identifying if the application to be developed or developed uses authentication or not
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def authentication(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will the system use authentication?**  ")
    else:
        print("  **The implemented system uses authentication ?**  ")
    print("")
    print("  1 - Yes ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q3"] = str(value)

    if value == 1:
        typeOfAuthentication(version)
    else:
        questions_and_answers["Q4"]="N/A"

"""
[Summary]: Method responsible for identifying the authentication scheme to be used or used by the application
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def typeOfAuthentication(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **What will be the authentication scheme to be implemented in the system?**  ")
    else:
        print("  **What is the authentication scheme implemented in the system ?**  ")

    print("  (This is a multiple choice question. Enter several options and end with 0.)  ")
    print("")

    print("  1 - Biometric-based authentication")
    print("  2 - Channel-based authentication ")
    print("  3 - Factors-based authentication")
    print("  4 - ID-based authentication ")
    print("  5 - Other")
    print("")

    while (1):
        value = validateInput(1, 6)
        if value == 0:
            return
        if value == 5:
            print("  Please specify the authentication scheme: (name between single quotes)  ")
            # TO-DO change this funtion input
            value2 = validateInput(2)
            questions_and_answers["Q4"] = questions_and_answers["Q4"] + str(value2) + ";"

        else:
            questions_and_answers["Q4"] = questions_and_answers["Q4"] + str(value) + ";"

"""
[Summary]: Method that allows identifying if the application to be developed or developed uses a database or not
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def hasDB(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will the system use a Database?**  ")
    else:
        print("  **Does the system use a Database?**  ")
    print("")
    print("  1 - Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q5"] = str(value)

    if value == 1:
        typeOfDatabase(version)
        if questions_and_answers["Q6"] == '1' or questions_and_answers["Q6"] == '2':
            whichDatabase(version)
        sensitiveData(version)
        storageLocation(version)
    else:
        questions_and_answers["Q6"] = "N/A"
        questions_and_answers["Q7"] = "N/A"
        questions_and_answers["Q8"] = "N/A"
        return

"""
[Summary]: Method to identify the type of storage of the end-users of the application
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def typeOfDatabase(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **What will be the type of data storage ?**  ")
    else:
        print("  **What is type of data storage?**  ")
    print("")
    print("  1 - SQL DBMS (Relational Database) ")
    print("  2 - NoSQL DBMS ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q6"] = str(value)

"""
[Summary]: Method allowing the identification of the DBMS to be used by the system
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def whichDatabase(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Which Database (DBMS) will be used ?**  ")
    else:
        print("  **What is the Database (DBMS) used?**  ")

    print("")

    if questions_and_answers["Q6"] == '1':
        print("  1 - SQL Server  ")
        print("  2 - MySQL  ")
        print("  3 - PostgreSQL  ")
        print("  4 - SQLite  ")
        print("  5 - OracleDB  ")
        print("  6 - MariaDB  ")

    if questions_and_answers["Q6"] == '2':
        print("  7 - Cassandra ")
        print("  8 - CosmosDB  ")
        print("  9 - MongoDB  ")
        print(" 10 - JADE      ")
        print(" 11 - HBase     ")
        print(" 12 - Neo4j     ")
        print(" 13 - Redis     ")
        print(" 14 - Other     ")
        print("")
    value = validateInput(1, 15)
    if value == 14:
        print("  Please specify the name of database: (name between single quotes)  ")
        value2 = validateInput(2)
        questions_and_answers["Q7"] = str(value2)
    else:
        questions_and_answers["Q7"] = str(value)

"""
[Summary]: Method allowing the identification of the type of data handled by the system (personal, confidential or critical)
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def sensitiveData(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **What type of information will the application handle?**  ")
    else:
        print("  **What is the type of information handled by the application?**  ")

    print("  (This is a multiple choice question. Enter several options and end with 0.)  ")
    print("")
    print("  1 - Personal Information (Names, Address,...)  ")
    print("  2 - Confidential Data  ")
    print("  3 - Critical Data  ")
    print("  4 - Other ")
    print("")

    while (1):
        value = validateInput(1, 5)
        if value == 0:
            return
        if value == 4:
            print("  Please specify the type of information handled: (name between single quotes)  ")
            # TO-DO change this funtion input
            value2 = validateInput(2)
            # question_5["4"] = str(value2)
            questions_and_answers["Q8"] = questions_and_answers["Q8"] + str(value2) + ";"
        else:
            questions_and_answers["Q8"] = questions_and_answers["Q8"] + str(value) + ";"

"""
[Summary]: Method allowing the identification of whether or not the application allows users to register
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def storageLocation(version):
    print("")
    print("---")
    print("")
    
    if version == 1:
        print("  **Where will the system store the data?**  ")
    else:
        print("  **What is the storage location of the system?**  ")
    print("")
    print("  1 - Local Storage (Centralized databse) ")
    print("  2 - Remote Storage (Cloud Database) ")
    print("  3 - Both (Hybrid Database) ")
    print("")

    value = validateInput(1, 4)
    questions_and_answers["Q9"] = str(value)

"""
[Summary]: Method allowing the identification of whether or not the application allows users to register
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def userRegist(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will there be a user registration?**  ")
    else:
        print("  **Is there a user registration?**  ")
    print("")
    print("  1 - Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q10"] = str(value)

    if value == 1:
        typeOfUserRegist(version)
    else:
        questions_and_answers["Q11"] = "N/A"

"""
[Summary]: Method to identify the type of registration of users to the system
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def typeOfUserRegist(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print(" **Which way of user registration will be used ?**  ")
    else:
        print(" **Which way of user registration it's used ?**  ")
    print("")
    print("  1 - The users will register themselves  ")
    print("  2 - Will be an administrator that will register the users  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q11"] = str(value)

"""
[Summary]: Method to identify the programming language to be used or used for coding the application
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def languages(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Which programming languages will be used in the implementation of the system ?**  ")
    else:
        print("  **What is the programming languages used in the implementation of the system ?**  ")

    print("  (This is a multiple choice question. Enter several options and end with 0.)  ")
    print("")
    print("  1 - C#  ")
    print("  2 - Obective-C/C/C++ ")
    print("  3 - HTML5 + CSS + JavaScript ")
    print("  4 - Java  ")
    print("  5 - Javascript ")
    print("  6 - PHP  ")
    print("  7 - Python  ")
    print("  8 - Ruby  ")
    print("  9 - Kotlin ")
    print(" 10 - Swift ")
    print(" 11 - Other/Property Language  ")
    print("")

    while (1):
        value = validateInput(1, 12)
        if value == 0:
            return
        if value == 9:
            print("  Please specify the programming language: (name between single quotes)  ")
            # TO-DO change this funtion input
            value2 = validateInput(2)
            # question_9["8"]  = str(value2)
            questions_and_answers["Q12"] = questions_and_answers["Q12"] + str(value2) + ";"

        else:
            questions_and_answers["Q12"] = questions_and_answers["Q12"] + str(value) + ";"

"""
[Summary]: Method to identify if the application uses or not user input forms
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def inputForms(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will the system allow user input forms?**  ")
    else:
        print("  **Has the system user input forms?**  ")
    print("")
    print("  1 -Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q13"] = str(value)

"""
[Summary]: Method to identify the application allows or not upload files
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def allowUploadFiles(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will the system allow upload files?**  ")
    else:
        print("  **Does the system allow upload files?**  ")
    print("")
    print("  1 - Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q14"] = str(value)

"""
[Summary]: Method to identify the application records or not logs
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def systemLogs(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will the system record event logs?**  ")
    else:
        print("  **Has The system a logging regist?**  ")
    print("")
    print("  1 - Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q15"] = str(value)

"""
[Summary]: Method to identify the application allows or not regular updates
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def allowUpdateSystem(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will the system allow regular updates?**  ")
    else:
        print("  **Has the system a regular updates?**  ")
    print("")
    print("  1 - Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q16"] = str(value)

"""
[Summary]: Method to identify the application uses or not third-party apps
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def allowThirdParty(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Will the system use third-party apps?**  ")
    else:
        print("  **Does the system use third-party apps?**  ")
    print("")
    print("  1 - Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q17"] = str(value)

"""
[Summary]: Method to identify the Cloud development model (environment) used by the application
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def cloudPlatform(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **What is the environment in which the app will be used?**  ")
    else:
        print("  **What is the environment in which the system is used?** ")
    print("  1 - Community Cloud (Remote connection) ")
    print("  2 - Hybrid Cloud (Mix connection) ")
    print("  3 - Private Cloud (Local connection) ")
    print("  4 - Public Cloud (Remote connection)")
    print("  5 - Virtual Private Cloud")
    print("")

    value = validateInput(1, 6)
    questions_and_answers["Q18"] = str(value)

"""
[Summary]: Method that allows identifying if the user wants to specify some hardware details (network and authentication scheme) or not
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def hardwareSpecs(version):
    print("")
    print("---")
    print("")
    print("  **Do you want to further specify hardware details concerning the system ?**  ")
    print("")
    print("  1 - Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q19"] = str(value)

    if value == 1:
        hardwareAuth(version)
        hardwareComunication(version)
    else:
        questions_and_answers["Q20"] = "N/A"
        questions_and_answers["Q21"] = "N/A"

"""
[Summary]: Method allowing the identification of the authentication paradigm implemented in relation to the hardware
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def hardwareAuth(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **What will be the type of authentication implemented in hardware?**  ")
    else:
        print("  **What is the type of authentication implemented in hardware?**  ")
    print("")
    print("  1 - No Authentication   ")
    print("  2 - Symmetric Key   ")
    print("  3 - Basic Authentication (user/pass)  ")
    print("  4 - Certificates (X.509)   ")
    print("  5 - TPM (Trusted Platform Module)  ")
    print("")

    value = validateInput(1, 6)
    questions_and_answers["Q20"] = str(value)

"""
[Summary]: Method allowing the identification of wireless network technologies implemented in relation to hardware
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def hardwareComunication(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **What wireless technologies will be on the hardware?**  ")
    else:
        print("  **What are the wireless tecnologies presents in hardware. Enter several options and end with 0.**  ")
    print("")
    print("  1 - 3G ")
    print("  2 - 4G / LTE ")
    print("  3 - 5G  ")
    print("  4 - GSM (2G)  ")
    print("  5 - Bluetooth  ")
    print("  6 - Wi-Fi  ")
    print("  7 - GPS  ")
    print("  8 - RFID  ")
    print("  9 - NFC  ")
    print("  10 - ZigBee ")
    print("  11 - Others")
    print("")
    while(1):
        value = validateInput(1, 12)
        if value == 0:
            break
        if value == 12:
            print("Please specify the wireless technologies: (name between single quotes)")
            value2 = validateInput(2)
            questions_and_answers["Q21"] = questions_and_answers["Q21"] + str(value2) + ";"
        else:
            questions_and_answers["Q21"] = questions_and_answers["Q21"] + str(value) + ";"

"""
[Summary]: Method to identify the existence or not of the possibility of physical access to the system (data centre and mobile device)
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def dataCenterAcess(version):
    print("")
    print("---")
    print("")
    if version == 1:
        print("  **Can someone gain physical access to the machine where the system will operates?** ")
    else:
        print("  **Can someone gain physical access to the machine where the system operates?**  ")
    print("")
    print("  1 -Yes  ")
    print("  2 - No  ")
    print("")

    value = validateInput(1, 3)
    questions_and_answers["Q22"] = str(value)

"""
[Summary]: Method to open/create and store in the file 'ans.txt' the answers to the user questionnaire
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def print_data():
    generate_file = open("ans.txt", "w")

    list_aux = []
    # it is a multiple question

    # find if the answer correspond to option "others" (means that is user input text) OR fix this buy make it simple, verify if it the answer has only letters xD
    # find if the first caracter is a letter and if the answer has no more options
    if questions_and_answers["Q1"][0].isdigit() == False and questions_and_answers["Q1"].find(";") == -1:
        list_aux.append(questions_and_answers["Q1"])

    else:

        # variable aux is a list that contains the answers chosen by the user to the question in cause
        # cut the string in the delimitator ";"
        aux = questions_and_answers["Q1"].split(";")

        # delete last item (= None)
        aux = aux[:-1]
        # print(aux)

        # iterate the answers chosen by the user
        for item in aux:

            # iterate the options of the question and check what the chosen answers match
            for option in question1:
                if item == option:
                    list_aux.append(question1[option])

            # case of user input text
            if item.isdigit() == False:
                list_aux.append(item)

    # print(list_aux)
    print("{:22} {:3} {:40} ".format("Mobile Platform", ":", ' ; '.join(list_aux)))
    table_for_report.append(["Mobile Platform", ' ; '.join(list_aux)])

    answers_list.append(questions_and_answers["Q1"])
    comments_list.append(' ; '.join(list_aux))

    for n in question2:
        item = questions_and_answers["Q2"]
        if item == n:
            print("{:22} {:3} {:40}".format("Application domain type", ":", question2[n]))

            table_for_report.append(["Application domain type", question2[n]])

            answers_list.append(questions_and_answers["Q2"])
            comments_list.append(question2[n])

    for n in question3:
        item = questions_and_answers["Q3"]
        if item == n:
            print("{:22} {:3} {:40}".format("Authentication", ":", question3[n]))

            table_for_report.append(["Authentication", question3[n]])

            answers_list.append(questions_and_answers["Q3"])
            comments_list.append(question3[n])

    list_aux = []
    # it is a multiple question
    # find if the answer correspond to option "others" (means that is user input text) or not answered
    if questions_and_answers["Q4"][0].isdigit() == False and questions_and_answers["Q4"].find(";") == -1:
        list_aux.append(questions_and_answers["Q4"])
    else:
        # variable aux is a list that contains the answers chosen by the user to the question in cause
        # cut the string in the delimitator ";"
        aux = questions_and_answers["Q4"].split(";")

        # delete last item (= None)
        aux = aux[:-1]

        for item in aux:
            for option in question4:
                if item == option:
                    list_aux.append(question4[option])

            # case of user input text
            if item.isdigit() == False:
                list_aux.append(item)

    print("{:22} {:3} {:40} ".format("Authentication schemes", ":", ' ; '.join(list_aux)))

    table_for_report.append(["Authentication schemes", ' ; '.join(list_aux)])

    answers_list.append(questions_and_answers["Q4"])
    comments_list.append(' ; '.join(list_aux))

    for n in question5:
        item = questions_and_answers["Q5"]
        if item == n:
            print("{:22} {:3} {:40} ".format("Has DB", ":", question5[n]))

            table_for_report.append(["Has DB", question5[n]])

            answers_list.append(questions_and_answers["Q5"])
            comments_list.append(question5[n])

    item = questions_and_answers["Q6"]
    # case this question is not answered, and the answer it will be "N/A"
    if questions_and_answers["Q6"].isdigit() == False:
        print("{:22} {:3} {:40} ".format("Type of database", ":", item))

        table_for_report.append(["Type of database", item])

        answers_list.append(questions_and_answers["Q6"])
        comments_list.append(item)

    else:
        for n in question6:
            if item == n:
                print("{:22} {:3} {:40} ".format("Type of database", ":", question6[n]))

                table_for_report.append(["Type of database", question6[n]])

                answers_list.append(questions_and_answers["Q6"])
                comments_list.append(question6[n])

    item = questions_and_answers["Q7"]
    for n in question7:
        if item == n:
            print("{:22} {:3} {:40} ".format("Which DB", ":", question7[n]))

            table_for_report.append(["Which DB", question7[n]])

            answers_list.append(questions_and_answers["Q7"])
            comments_list.append(question7[n])

    # case of user input text
    if item.isdigit() == False:
        print("{:22} {:3} {:40} ".format("Which DB", ":", item))

        table_for_report.append(["Which DB", item])

        answers_list.append(questions_and_answers["Q7"])
        comments_list.append(item)

    list_aux = []
    # it is a multiple question

    # find if the answer correspond to option "others" (means that is user input text) or not answered
    if questions_and_answers["Q8"][0].isdigit() == False and questions_and_answers["Q8"].find(";") == -1:
        list_aux.append(questions_and_answers["Q8"])
    else:

        # variable aux is a list that contains the answers chosen by the user to the question in cause
        # cut the string in the delimitator ";"
        aux = questions_and_answers["Q8"].split(";")

        # delete last item (= None)
        aux = aux[:-1]

        for item in aux:
            for option in question8:
                if item == option:
                    list_aux.append(question8[option])
            # case of user input text
            if item.isdigit() == False:
                list_aux.append(item)

    print("{:22} {:3} {:40} ".format("Type of information handled", ":", ' ; '.join(list_aux)))

    table_for_report.append(["Type of information handled", ' ; '.join(list_aux)])

    answers_list.append(questions_and_answers["Q8"])
    comments_list.append(' ; '.join(list_aux))

    item = questions_and_answers["Q9"]
    # case this question is not answered, and the answer it will be "N/A"
    if questions_and_answers["Q9"].isdigit() == False:
        print("{:22} {:3} {:40} ".format("Storage Location", ":", item))

        table_for_report.append(["Storage Location", item])

        answers_list.append(questions_and_answers["Q9"])
        comments_list.append(item)

    else:
        for n in question9:
            if item == n:
                print("{:22} {:3} {:40} ".format("Storage Location", ":", question9[n]))

                table_for_report.append(["Storage Location", question9[n]])

                answers_list.append(questions_and_answers["Q9"])
                comments_list.append(question9[n])

    for n in question10:
        item = questions_and_answers["Q10"]
        if item == n:
            print("{:22} {:3} {:40}".format("User Registration", ":", question10[n]))

            table_for_report.append(["User Registration", question10[n]])

            answers_list.append(questions_and_answers["Q10"])
            comments_list.append(question10[n])

    item = questions_and_answers["Q11"]
    if questions_and_answers["Q11"].isdigit() == False:
        print("{:22} {:3} {:40} ".format("Type of Registration", ": ", item))

        table_for_report.append(["Type of Registration", item])

        answers_list.append(questions_and_answers["Q11"])
        comments_list.append(item)
    else:
        for n in question11:
            if item == n:
                print("{:22} {:3} {:40} ".format("Type of Registration", ": ", question11[n]))

                table_for_report.append(["Type of Registration", question11[n]])

                answers_list.append(questions_and_answers["Q11"])
                comments_list.append(question11[n])

    list_aux = []
    # it is a multiple question

    # find if the answer correspond to option "others" (means that is only user input text)
    if questions_and_answers["Q12"][0].isdigit() == False and questions_and_answers["Q12"].find(";") == -1:
        list_aux.append(questions_and_answers["Q12"])
    else:

        # cut the string in the delimitator ";"
        aux = questions_and_answers["Q12"].split(";")

        # delete last item (= None)
        aux = aux[:-1]

        for item in aux:
            for option in question12:
                if item == option:
                    list_aux.append(question12[option])

            # case of user input text
            if item.isdigit() == False:
                list_aux.append(item)

    print("{:22} {:3} {:40} ".format("Programming Languages", ":", ' ; '.join(list_aux)))

    table_for_report.append(["Programming Languages", ' ; '.join(list_aux)])

    answers_list.append(questions_and_answers["Q12"])
    comments_list.append(' ; '.join(list_aux))

    for n in question13:
        item = questions_and_answers["Q13"]
        if item == n:
            print("{:22} {:3} {:40} ".format("Input Forms", ":", question13[n]))

            table_for_report.append(["Input Forms", question13[n]])

            answers_list.append(questions_and_answers["Q13"])
            comments_list.append(question13[n])

    for n in question14:
        item = questions_and_answers["Q14"]
        if item == n:
            print("{:22} {:3} {:40} ".format("Upload Files", ":", question14[n]))

            table_for_report.append(["Upload Files", question14[n]])

            answers_list.append(questions_and_answers["Q14"])
            comments_list.append(question14[n])

    for n in question15:
        item = questions_and_answers["Q15"]
        if item == n:
            print("{:22} {:3} {:40} ".format("The system has logs", ":", question15[n]))

            table_for_report.append(["The system has logs", question15[n]])

            answers_list.append(questions_and_answers["Q15"])
            comments_list.append(question15[n])

    for n in question16:
        item = questions_and_answers["Q16"]
        if item == n:
            print("{:22} {:3} {:40} ".format("The system has regular updates", ":", question16[n]))

            table_for_report.append(["The system has regular updates", question16[n]])

            answers_list.append(questions_and_answers["Q16"])
            comments_list.append(question16[n])

    for n in question17:
        item = questions_and_answers["Q17"]
        if item == n:
            print("{:22} {:3} {:40} ".format("The system has third-party", ":", question17[n]))

            table_for_report.append(["The system has third-party", question17[n]])

            answers_list.append(questions_and_answers["Q17"])
            comments_list.append(question17[n])

    for n in question18:
        item = questions_and_answers["Q18"]
        if item == n:
            print("{:22} {:3} {:40}".format("System Cloud Environments", ":", question18[n]))

            table_for_report.append(["System Cloud Environments", question18[n]])

            answers_list.append(questions_and_answers["Q18"])
            comments_list.append(question18[n])

    for n in question19:
        item = questions_and_answers["Q18"]
        if item == n:
            print("{:22} {:3} {:40} ".format("Hardware Specification", ":", question19[n]))

            table_for_report.append(["Hardware Specification", question19[n]])

            answers_list.append(questions_and_answers["Q19"])
            comments_list.append(question19[n])

    for n in question20:
        item = questions_and_answers["Q20"]
        if item == n:
            print("{:22} {:3} {:40} ".format("HW Authentication", ":", question20[n]))

            table_for_report.append(["HW Authentication", question20[n]])

            answers_list.append(questions_and_answers["Q20"])
            comments_list.append(question20[n])

    list_aux = []
    # it is a multiple question

    # find if the answer correspond to option "others" (means that is only user input text)
    if questions_and_answers["Q21"][0].isdigit() == False and questions_and_answers["Q21"].find(";") == -1:
        list_aux.append(questions_and_answers["Q21"])
    else:
        # cut the string in the delimitator ";"
        aux = questions_and_answers["Q21"].split(";")

        # delete last item (= None)
        aux = aux[:-1]

        for item in aux:
            for option in question21:
                if item == option:
                    list_aux.append(question21[option])
            # case of user input text
            if item.isdigit() == False:
                list_aux.append(item)

    print("{:22} {:3} {:40} ".format("HW Wireless Tech", ":", ' ; '.join(list_aux)))

    table_for_report.append(["HW Wireless Tech", ' ; '.join(list_aux)])

    answers_list.append(questions_and_answers["Q21"])
    comments_list.append(' ; '.join(list_aux))

    for n in question22:
        item = questions_and_answers["Q22"]
        if item == n:
            print("{:22} {:3} {:40} ".format("Device or Data Center Physical Access", ":", question22[n]))

            table_for_report.append(["Device or Data Center Physical Access", question22[n]])

            answers_list.append(questions_and_answers["Q22"])
            comments_list.append(question22[n])

    # write / generate a file with all answers
    for i in range(0, len(answers_list)):
        generate_file.write("{:20}{:3}{:20}\n".format(answers_list[i], " # ", comments_list[i]))

    generate_file.close()

"""
[Summary]: Method to convert the markdown Security Requirements report to html and pdf format
[Arguments]: No arguments
[Returns]: No return
"""
def requirements_convert_report():
    # input_filename = ("guides/example_report.md")
    # input_filename = "some_markdown.md")
    input_filename = ("SECURITY_REQUIREMENTS.md")

    output_filename = ("SECURITY_REQUIREMENTS.html")

    with open(input_filename, "r") as f:
        html_text = markdown(f.read(), extensions=['markdown.extensions.tables', 'markdown.extensions.sane_lists'])

    out = open(output_filename, "w")
    out.write(html_text)

    # writing in pdf file, the html content

    resultFile = open("SECURITY_REQUIREMENTS.pdf", "w+b")
    pisa.CreatePDF(html_text, dest=resultFile)

"""
[Summary]: Method to convert the markdown Security Best Practices Guidelines (SBPG) report to html and pdf format
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def security_best_practices_convert_report():
    # input_filename = ("guides/example_report.md")
    # input_filename = "some_markdown.md")
    input_filename = ("GOOD_PRACTICES.md")

    output_filename = ("GOOD_PRACTICES.html")

    with open(input_filename, "r") as f:
        html_text = markdown(f.read(), extensions=['markdown.extensions.tables', 'markdown.extensions.sane_lists'])

    out = open(output_filename, "w")
    out.write(html_text)

    # writing in pdf file, the html content

    resultFile = open("GOOD_PRACTICES.pdf", "w+b")
    pisa.CreatePDF(html_text, dest=resultFile)


"""
[Summary]: Method to convert the markdown Security Mechanism Elicitation (SME) report to html and pdf format
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def mechanisms_convert_report():
    # input_filename = ("guides/example_report.md")
    # input_filename = "some_markdown.md")
    input_filename = ("SECURITY_MECHANISMS.md")

    output_filename = ("SECURITY_MECHANISMS.html")

    with open(input_filename, "r") as f:
        html_text = markdown(f.read(), extensions=['markdown.extensions.tables', 'markdown.extensions.sane_lists'])

    out = open(output_filename, "w")
    out.write(html_text)

    # writing in pdf file, the html content

    resultFile = open("SECURITY_MECHANISMS.pdf", "w+b")
    pisa.CreatePDF(html_text, dest=resultFile)

"""
[Summary]: Method to convert the markdown Attack Models Elicitation (AME) report to html and pdf format
[Arguments]: 
    - $version$: An integer constant equal to unity
[Returns]: No return
"""
def attack_models_convert_report():
    # input_filename = ("guides/example_report.md")
    # input_filename = "some_markdown.md")
    input_filename = ("ATTACKS_MAPPING.md")

    output_filename = ("ATTACKS_MAPPING.html")

    with open(input_filename, "r") as f:
        html_text = markdown(f.read(), extensions=['markdown.extensions.tables', 'markdown.extensions.sane_lists'])

    out = open(output_filename, "w")
    out.write(html_text)

    # writing in pdf file, the html content

    resultFile = open("ATTACKS_MAPPING.pdf", "w+b")
    pisa.CreatePDF(html_text, dest=resultFile)

"""
[Summary]: Method to convert the markdown Security Test Specification and Automation Tools (STSAT) report to html and pdf format
[Arguments]: No arguments
[Returns]: No return
"""
def security_test_recommendation_convert_report():
    # input_filename = ("guides/example_report.md")
    # input_filename = "some_markdown.md")
    input_filename = ("TEST_SPECIFICATION.md")

    output_filename = ("TEST_SPECIFICATION.html")

    with open(input_filename, "r") as f:
        html_text = markdown(f.read(), extensions=['markdown.extensions.tables', 'markdown.extensions.sane_lists'])

    out = open(output_filename, "w")
    out.write(html_text)

    # writing in pdf file, the html content

    resultFile = open("TEST_SPECIFICATION.pdf", "w+b")
    pisa.CreatePDF(html_text, dest=resultFile)

"""
[Summary]: Method auxiliary to the input method responsible for processing user requests
[Arguments]: No arguments
[Returns]: No return
"""
def switch1():
    val = 0
    while True:
        try:
            val = int(input("\nWhat is your option?\n"))
            print("-->")
        except ValueError:
            print("Error! Enter a whole number between 1 and 8, according to the menu above!")
            continue
        
        if val>8 or val<1:
            print("Wrong! Enter a whole number between 1 and 8, according to the menu above!")
            continue
        else:
            with Switch(val) as case:
                if case(1):
                    print("---")
                    print("")
                    print("  **Which way do you want to run this tool?**  ")
                    print("")
                    print("  1 - Answer the questions one by one.  ")
                    print("  2 - Use a text file with the answers already written.  ")
                    print("")
                    
                    input_choice = validateInput(1, 3)
                    print("")
                    
                    # answer the questions by hand
                    if input_choice == 1:
                        arqui(version)
                        domain(version)
                        authentication(version)
                        hasDB(version)
                        userRegist(version)
                        languages(version)
                        inputForms(version)
                        allowUploadFiles(version)
                        systemLogs(version)
                        allowUpdateSystem(version)
                        allowThirdParty(version)
                        cloudPlatform(version)
                        hardwareSpecs(version)
                        dataCenterAcess(version)
                        print("**The questionnaire is over!**")
                    
                    # answers already written in the input file
                    else:
                        print("---")
                        print("")
                        print("  **What is the name of the input file (ans.txt)?**  ")
                        print("")
                        readInputFromFile()
                    
                        questions_and_answers["Q1"] = input_list[0]
                        questions_and_answers["Q2"] = input_list[1]
                        questions_and_answers["Q3"] = input_list[2]
                        questions_and_answers["Q4"] = input_list[3]
                        questions_and_answers["Q5"] = input_list[4]
                        questions_and_answers["Q6"] = input_list[5]
                        questions_and_answers["Q7"] = input_list[6]
                        questions_and_answers["Q8"] = input_list[7]
                        questions_and_answers["Q9"] = input_list[8]
                        questions_and_answers["Q10"] = input_list[9]
                        questions_and_answers["Q11"] = input_list[10]
                        questions_and_answers["Q12"] = input_list[11]
                        questions_and_answers["Q13"] = input_list[12]
                        questions_and_answers["Q14"] = input_list[13]
                        questions_and_answers["Q15"] = input_list[14]
                        questions_and_answers["Q16"] = input_list[15]
                        questions_and_answers["Q17"] = input_list[16]
                        questions_and_answers["Q18"] = input_list[17]
                        questions_and_answers["Q19"] = input_list[18]
                        questions_and_answers["Q20"] = input_list[19]
                        questions_and_answers["Q21"] = input_list[20]
                        questions_and_answers["Q22"] = input_list[21]
                    
                        information_capture()
                    
                        print("# Processing Done! Choose another option to process the requests!")
                    
                if case(2):
                    print("\n********************************************************************************************\n")
                    print("\t\t The request for security requirements is in progress ... \n\n")
                    get_requirements()
                    webbrowser.open_new(r'file:///Users/FranciscoChimuco/SecD4CLOUDMOBILEv1.2/SECURITY_REQUIREMENTS.pdf')
                    information_capture()
                    
                if case(3):
                    print("\n********************************************************************************************\n")
                    print("\t\t The request for best practice guidelines is in progress ... \n\n")
                    get_security_best_practices()
                    webbrowser.open_new(r'file:///Users/FranciscoChimuco/SecD4CLOUDMOBILEv1.2/GOOD_PRACTICES.pdf')
                    information_capture()
                    
                if case(4):
                    print("\n********************************************************************************************\n")
                    print("\t\t The request for security mechanisms is in progress ... \n\n")
                    get_mechanisms()
                    webbrowser.open_new(r'file:///Users/FranciscoChimuco/SecD4CLOUDMOBILEv1.2/SECURITY_MECHANISMS.pdf')
                    information_capture()
                    
                if case(5):
                    print("\n********************************************************************************************\n")
                    print("\t\t The request for attack models is in progress ... \n\n")
                    get_attack_models()
                    webbrowser.open_new(r'file:///Users/FranciscoChimuco/SecD4CLOUDMOBILEv1.2/ATTACKS_MAPPING.pdf')
                    information_capture()
                    
                if case(6):
                    print("\n********************************************************************************************\n")
                    print("\t\t The request for the security testing specification and tools is in progress ... \n\n")
                    get_security_test_recommendation()
                    webbrowser.open_new(r'file:///Users/FranciscoChimuco/SecD4CLOUDMOBILEv1.2/TEST_SPECIFICATION.pdf')
                    information_capture()
                    
                if case(7):
                    print("\n********************************************************************************************\n")
                    print("\t\t The full report request is in progress ... \n\n")
                    fullReport()
                    webbrowser.open_new(r'file:///Users/FranciscoChimuco/SecD4CLOUDMOBILEv1.2/FULL_REPORT.pdf')
                    information_capture()
                    
                if case(8):
                    print("\n\n*** Application finished successfully! ***\n\n")
                    exit(0)
                    
                if case.default:
                    print("\nError! Insert a valid value between 1 and 8!\n")

        break

"""
[Summary]: Method responsible for creating the main menu and capturing the user data
[Arguments]: No argument
[Return]: No return
"""
def information_capture():
    print("************************************************************************************************")
    print("\nWelcome to SecD4CLOUDMOBILE Framework!\n")
    print("\nWhat would you like to do?\n")
    print("\n1. First, Answer the Questions Necessary for Possible Processing")
    print("\n2. Security Requirement Elicitation Request")
    print("\n3. Secure Development Best Practice Guidelines Request")
    print("\n4. Secure Development Security Mechanisms Request")
    print("\n5. Attack Model Mapping Request")
    print("\n6. Security Test Specification and Tool Request")
    print("\n7. Full Report Request")
    print("\n8. Exit")
    print("\n\nSelect your option (1-8):")
    switch1()

"""
[Summary]: Method responsible for processing information about CSRE
[Arguments]: No arguments
[Return]: No return
"""
def get_requirements():
    print("")
    print("  Processing information.....")
    print("")

    print_data()

    report = open("SECURITY_REQUIREMENTS.md", "w")
    report.write("# Final Security Requirements Report " + '\n')
    report.write("\n")

    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", "", "|", "", "|"))
    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", ":--------", "|", ":---------", "|"))

    '''
    for i in range( 0,len(table_for_report) ):
        report.write("| " + table_for_report[i][0] + " | " + table_for_report[i][1] + " | \n" )
    '''
    for i in range(0, len(table_for_report)):
        report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", table_for_report[i][0], "|", table_for_report[i][1], "|"))

    report.write("\n")

    # confidentiality requirement
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        confidentiality = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Confidentiality requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_confidentiality = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee confidentiality requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/confidentiality.md", "w")
        f.write("# Confidentiality \n\n")
        f.write(confidentiality.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/confidentiality.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_confidentiality.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/confidentiality.md", "r").read())

    # integrity requirement
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        integrity = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Integrity requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_integrity = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee integrity requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/integrity.md", "w")
        f.write("# Integrity \n\n")
        f.write(integrity.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/integrity.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_integrity.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/integrity.md", "r").read())

    # availability requirement
    report.write("\n")
    report.write("\n")

    availability = openai.Completion.create(
        model="text-davinci-003",
        prompt="What is Availability requirement in security engineering? Generate in Markdown format. \n\n",
        max_tokens=200,
        n=1,
        best_of=2,
    )

    feilure_availability = openai.Completion.create(
        model="text-davinci-003",
        prompt="If we feilure to guarantee availability requirement, what happened to system? Generate in Markdown format. \n\n",
        max_tokens=200,
        n=1,
        best_of=2,
    )

    f = open("security_requirements/availability.md", "w")
    f.write("# Availability \n\n")
    f.write(availability.choices[0]["text"].strip())
    f.close()

    f = open("security_requirements/availability.md", "a")
    f.write("\n\n")
    f.write("## Warning: \n\n")
    f.write(feilure_availability.choices[0]["text"].strip())
    f.close()

    report.write(open("security_requirements/availability.md", "r").read())

    # authentication requirement
    if questions_and_answers["Q3"].find("1") != -1:
        report.write("\n")
        report.write("\n")

        authentication = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Authentication requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_authentication = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee authentication requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/authentication.md", "w")
        f.write("# Authentication \n\n")
        f.write(authentication.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/authentication.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_authentication.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/authentication.md", "r").read())

    # authorization requirement
    if questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            authorization = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Authorization requirement in security engineering? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            feilure_authorization = openai.Completion.create(
                model="text-davinci-003",
                prompt="If we feilure to guarantee authorization requirement, what happened to system? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            f = open("security_requirements/authorization.md", "w")
            f.write("# Authorization \n\n")
            f.write(authorization.choices[0]["text"].strip())
            f.close()

            f = open("security_requirements/authorization.md", "a")
            f.write("\n\n")
            f.write("## Warning: \n\n")
            f.write(feilure_authorization.choices[0]["text"].strip())
            f.close()

            report.write(open("security_requirements/authorization.md", "r").read())

    # non-repudiaton requirements
    if questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            non_repudiation = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Non-repudiation requirement in security engineering? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            feilure_non_repudiation = openai.Completion.create(
                model="text-davinci-003",
                prompt="If we feilure to guarantee non non-repudiation requirement, what happened to system? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            f = open("security_requirements/nonRepudiation.md", "w")
            f.write("# Non-repudiation \n\n")
            f.write(non_repudiation.choices[0]["text"].strip())
            f.close()

            f = open("security_requirements/nonRepudiation.md", "a")
            f.write("\n\n")
            f.write("## Warning: \n\n")
            f.write(feilure_non_repudiation.choices[0]["text"].strip())
            f.close()

            report.write(open("security_requirements/nonRepudiation.md", "r").read())

    # accountability requirements
    if questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")
            accountability = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Accountability requirement in security engineering? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            feilure_accountability = openai.Completion.create(
                model="text-davinci-003",
                prompt="If we feilure to guarantee accountability requirement, what happened to system? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            f = open("security_requirements/accountability.md", "w")
            f.write("# Accountability \n\n")
            f.write(accountability.choices[0]["text"].strip())
            f.close()

            f = open("security_requirements/accountability.md", "a")
            f.write("\n\n")
            f.write("## Warning: \n\n")
            f.write(feilure_accountability.choices[0]["text"].strip())
            f.close()

            report.write(open("security_requirements/accountability.md", "r").read())

    # reliability requirement
    report.write("\n")
    report.write("\n")

    reliability = openai.Completion.create(
        model="text-davinci-003",
        prompt="What is Reliability requirement in security engineering? Generate in Markdown format. \n\n",
        max_tokens=200,
        n=1,
        best_of=2,
    )

    feilure_reliability = openai.Completion.create(
        model="text-davinci-003",
        prompt="If we feilure to guarantee reliability requirement, what happened to system? Generate in Markdown format. \n\n",
        max_tokens=200,
        n=1,
        best_of=2,
    )

    f = open("security_requirements/reliability.md", "w")
    f.write("# Reliability \n\n")
    f.write(reliability.choices[0]["text"].strip())
    f.close()

    f = open("security_requirements/reliability.md", "a")
    f.write("\n\n")
    f.write("## Warning: \n\n")
    f.write(feilure_reliability.choices[0]["text"].strip())
    f.close()

    report.write(open("security_requirements/reliability.md", "r").read())

    # privacy security_requirements
    # if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
    #     report.write("\n")
    #     report.write("\n")
    #     report.write(open("security_requirements/privacy.md", "r").read())

    # physical security requirement
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1) and questions_and_answers["Q22"].find("1") != -1:
        report.write("\n")
        report.write("\n")

        physical_security = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Physical Security requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_physical_security = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee physical security requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/physicalSecurity.md", "w")
        f.write("# Physical Security \n\n")
        f.write(physical_security.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/physicalSecurity.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_physical_security.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/physicalSecurity.md", "r").read())

    # forgery resistance requirement
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1) and questions_and_answers["Q22"].find("1") != -1:
        report.write("\n")
        report.write("\n")

        forgery_resistence = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Forgery Resistence requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_forgery_resistence = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee forgery resistence requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/forgeryResistance.md", "w")
        f.write("# Forgery Resistence \n\n")
        f.write(forgery_resistence.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/forgeryResistance.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_forgery_resistence.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/forgeryResistance.md", "r").read())

    # tampering detection requirement
    if questions_and_answers["Q22"].find("1") != -1:
        report.write("\n")
        report.write("\n")

        tamper_detection = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Tamper Detection requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_tamper_detection = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee tamper detection requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/tamperDetection.md", "w")
        f.write("# Tamper Detection \n\n")
        f.write(tamper_detection.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/tamperDetection.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_tamper_detection.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/tamperDetection.md", "r").read())

    # data freshness requirement
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        data_freshness = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Data Freshness requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_data_freshness = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee data freshness requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/dataFreshness.md", "w")
        f.write("# Data Freshness \n\n")
        f.write(data_freshness.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/dataFreshness.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_data_freshness.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/dataFreshness.md", "r").read())

    # confinement requirement
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q14"].find("1") != -1 or questions_and_answers["Q16"].find("1") != -1 or questions_and_answers["Q17"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            confinement = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Confinement requirement in security engineering? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            feilure_confinement = openai.Completion.create(
                model="text-davinci-003",
                prompt="If we feilure to guarantee confinement requirement, what happened to system? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            f = open("security_requirements/confinement.md", "w")
            f.write("# Confinement \n\n")
            f.write(confinement.choices[0]["text"].strip())
            f.close()

            f = open("security_requirements/confinement.md", "a")
            f.write("\n\n")
            f.write("## Warning: \n\n")
            f.write(feilure_confinement.choices[0]["text"].strip())
            f.close()

            report.write(open("security_requirements/confinement.md", "r").read())

    # Interoperability requirement
    if questions_and_answers["Q14"].find("1") != -1 and questions_and_answers["Q16"].find("1") != -1 and questions_and_answers["Q17"].find("1") != -1:
        report.write("\n")
        report.write("\n")

        interoperability = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Interoperability requirement in security engineering? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        feilure_interoperability = openai.Completion.create(
            model="text-davinci-003",
            prompt="If we feilure to guarantee interoperability requirement, what happened to system? Generate in Markdown format. \n\n",
            max_tokens=200,
            n=1,
            best_of=2,
        )

        f = open("security_requirements/interoperability.md", "w")
        f.write("# Interoperability \n\n")
        f.write(interoperability.choices[0]["text"].strip())
        f.close()

        f = open("security_requirements/interoperability.md", "a")
        f.write("\n\n")
        f.write("## Warning: \n\n")
        f.write(feilure_interoperability.choices[0]["text"].strip())
        f.close()

        report.write(open("security_requirements/interoperability.md", "r").read())

    # data origin authentication
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q5"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            data_origin_auth = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Data Origin Authentication requirement in security engineering? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            feilure_data_origin_auth = openai.Completion.create(
                model="text-davinci-003",
                prompt="If we feilure to guarantee data origin authentication requirement, what happened to system? Generate in Markdown format. \n\n",
                max_tokens=200,
                n=1,
                best_of=2,
            )

            f = open("security_requirements/dataOriginAuthentication.md", "w")
            f.write("# Data Origin Authentication \n\n")
            f.write(data_origin_auth.choices[0]["text"].strip())
            f.close()

            f = open("security_requirements/dataOriginAuthentication.md", "a")
            f.write("\n\n")
            f.write("## Warning: \n\n")
            f.write(feilure_data_origin_auth.choices[0]["text"].strip())
            f.close()

            report.write(open("security_requirements/dataOriginAuthentication.md", "r").read())

    report.close()
    requirements_convert_report()
    print("\n\n # Processing done! Check your requirements in the SECURITY_REQUIREMENTS.pdf file")

"""
[Summary]: Method responsible for processing information about CSBPG module
[Arguments]: No arguments
[Return]: No return
"""
def get_security_best_practices():
    print("")
    print("  Processing information.....")
    print("")

    report = open("GOOD_PRACTICES.md", "w")
    report.write("# Final Security Good Practices " + '\n')
    report.write("\n")

    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", "", "|", "", "|"))
    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", ":--------", "|", ":---------", "|"))

    '''
    for i in range( 0,len(table_for_report) ):
        report.write("| " + table_for_report[i][0] + " | " + table_for_report[i][1] + " | \n" )
    '''
    for i in range(0, len(table_for_report)):
        report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", table_for_report[i][0], "|", table_for_report[i][1], "|"))

    report.write("\n")

    # IoT System security best practices guidelines
    if (questions_and_answers["Q1"].find("1") != -1 or questions_and_answers["Q1"].find("2") != -1 or questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("5") != -1 or questions_and_answers["Q1"].find("6") != -1):
        if questions_and_answers["Q1"].find("7") != -1:
            report.write("\n")
            report.write("\n")

            iot_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for IoT System development in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/IoT_Security_guide.md.md", "w")
            f.write("# Security Best Practices Guidelines for IoT System \n\n")
            f.write(iot_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/IoT_Security_guide.md", "r").read())

    # SQL Injection Prevention security best practices guidelines
    if questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4"):
        if questions_and_answers["Q5"].find("1") != -1 and questions_and_answers["Q6"].find("1"):
            report.write("\n\n")

            sqli_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for SQL Injection in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/SQL_Injection_Prevention_Cheat_Sheet.md.md", "w")
            f.write("# Security Best Practices Guidelines for SQL Injection \n\n")
            f.write(sqli_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/SQL_Injection_Prevention_Cheat_Sheet.md", "r").read())
    
    # Authentication security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1):
        report.write("\n\n")
        authentication_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Authentication in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Authentication_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Authentication \n\n")
        f.write(authentication_sbpg.choices[0]["text"].strip())
        f.close()
        report.write(open("security_best_practices_guidelines/Authentication_Cheat_Sheet.md", "r").read())
        
        # Multifactor authentication security best practices guidelines 
        if (questions_and_answers["Q4"].find("3") != -1):
            report.write("\n\n")

            multifactor_auth_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for Multifactor Authentication in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/Multifactor_Authentication_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for Multifactor Authentication \n\n")
            f.write(multifactor_auth_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/Multifactor_Authentication_Cheat_Sheet.md", "r").read())

    
    # Authorization security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            report.write("\n\n")
            authorization_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for Authorization in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/Authorization_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for Authorization \n\n")
            f.write(authorization_sbpg.choices[0]["text"].strip())
            f.close()
            report.write(open("security_best_practices_guidelines/Authorization_Cheat_Sheet.md", "r").read())
        
        # Transaction Authorization security best practices guidelines
        if (questions_and_answers["Q2"].find("3") != -1 or questions_and_answers["Q2"].find("6") != -1):
            report.write("\n\n")

            transaction_author_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for Transaction Authorization in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/Transaction_Authorization_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for Transaction Authorization \n\n")
            f.write(transaction_author_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/Transaction_Authorization_Cheat_Sheet.md", "r").read())
    
    # Cross-Site-Scripting security best practices guidelines
    if (questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1):
        if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
            if (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
                report.write("\n\n")

                xss_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for XSS in Markdown format. \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/Cross_Site_Scripting_Prevention_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for XSS \n\n")
                f.write(xss_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/Cross_Site_Scripting_Prevention_Cheat_Sheet.md", "r").read())
    
    # Cross Site Request Forgery security best practices guidelines
    if (questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1):
        if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
            if (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
                report.write("\n\n")

                csrf_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for CSRF in Markdown format. \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/Cross_Site_Request_Forgery_Prevention_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for CSRF \n\n")
                f.write(csrf_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/Cross_Site_Request_Forgery_Prevention_Cheat_Sheet.md", "r").read())
    
    # Cryptographic Storage security best practices guidelines
    if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n\n")

        cryptography_storage_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Cryptographic Storage in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Cryptographic_Storage_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Cryptographic Storage \n\n")
        f.write(cryptography_storage_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Cryptographic_Storage_Cheat_Sheet.md", "r").read())
    
    # Database security best practices guidelines    
    if (questions_and_answers["Q5"].find("1") != -1):
        report.write("\n\n")

        database_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Database_Security in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Database_Security_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Database Security \n\n")
        f.write(database_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Database_Security_Cheat_Sheet.md", "r").read())
          
    # Denial of Service security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        report.write("\n\n")

        dos_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Denial of Service in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Denial_of_Service_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Denial of Service \n\n")
        f.write(dos_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Denial_of_Service_Cheat_Sheet.md", "r").read())
        
    # File uploading security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            if (questions_and_answers["Q14"].find("1") != -1):
                report.write("\n\n")

                fileupload_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for File Upload in Markdown format. \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/File_Upload_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for File Upload \n\n")
                f.write(fileupload_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/File_Upload_Cheat_Sheet.md", "r").read())

    # HTML5 security best practices guidelines
    if (questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1):
        if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
            if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
                if (questions_and_answers["Q12"].find("3") != -1):
                    report.write("\n\n")

                    html5_sbpg = openai.Completion.create(
                        model="text-davinci-003",
                        prompt="Generate the security best practices guidelines for HTML5 Security in Markdown format. \n\n",
                        max_tokens=500,
                        n=1,
                        best_of=2,
                    )

                    f = open("security_best_practices_guidelines/HTML5_Security_Cheat_Sheet.md", "w")
                    f.write("# Security Best Practices Guidelines for HTML5 Security \n\n")
                    f.write(html5_sbpg.choices[0]["text"].strip())
                    f.close()

                    report.write(open("security_best_practices_guidelines/HTML5_Security_Cheat_Sheet.md", "r").read())
    
    # Securing CSS security best practices guidelines
    if (questions_and_answers["Q12"].find("3") != -1):
        report.write("\n\n")

        css_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Securing CSS in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Securing_Cascading_Style_Sheets_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Securing CSS \n\n")
        f.write(css_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Securing_Cascading_Style_Sheets_Cheat_Sheet.md", "r").read())
    
    # Injection security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            if (questions_and_answers["Q13"].find("1") != -1):
                report.write("\n\n")

                injection_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for Injection Prevention in Markdown format. \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/Injection_Prevention_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for Injection Prevention \n\n")
                f.write(injection_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/Injection_Prevention_Cheat_Sheet.md", "r").read())
    
    # Injection prevention in Java security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            if (questions_and_answers["Q12"].find("4") != -1):
                report.write("\n\n")

                injection_java_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for Injection Prevention in Java in Markdown format. \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/Injection_Prevention_in_Java_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for Injection Prevention in Java \n\n")
                f.write(injection_java_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/Injection_Prevention_in_Java_Cheat_Sheet.md", "r").read())
    
    # Secure Logging security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            if (questions_and_answers["Q15"].find("1") != -1):
                report.write("\n\n")

                logging_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for Logging in Markdown format. \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/Logging_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for Logging \n\n")
                f.write(logging_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/Logging_Cheat_Sheet.md", "r").read())
    
    # Logging Vocabulary security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            if (questions_and_answers["Q15"].find("1") != -1):
                report.write("\n\n")

                logging_vocabulary_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for Logging Vocabulary in Markdown format. \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/Logging_Vocabulary_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for Logging Vocabulary \n\n")
                f.write(logging_vocabulary_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/Logging_Vocabulary_Cheat_Sheet.md", "r").read())
    
    # Nodejs security best practices guidelines 
    if (questions_and_answers["Q12"].find("5") != -1):
        report.write("\n\n")

        nodejs_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Nodejs Security in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Nodejs_Security_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Nodejs Security \n\n")
        f.write(nodejs_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Nodejs_Security_Cheat_Sheet.md", "r").read())
    
    # NPM security best practices guidelines 
    if (questions_and_answers["Q12"].find("5") != -1):
        report.write("\n\n")

        npm_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for NPM Security in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/NPM_Security_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for NPM Security \n\n")
        f.write(npm_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/NPM_Security_Cheat_Sheet.md", "r").read())
    
    # Third Party Javascript security best practices guidelines 
    if (questions_and_answers["Q12"].find("5") != -1):
        if (questions_and_answers["Q17"].find("1") != -1):
            report.write("\n\n")

            tpjm_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for Third-Party Javascript Management in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/Third_Party_Javascript_Management_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for Third-Party Javascript Management \n\n")
            f.write(tpjm_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/Third_Party_Javascript_Management_Cheat_Sheet.md", "r").read())
            
    # Password Storage security best practices guidelines
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            report.write("\n\n")

            password_storage_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for Password Storage in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/Password_Storage_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for Password Storage \n\n")
            f.write(password_storage_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/Password_Storage_Cheat_Sheet.md", "r").read())
    
    # PHP Configuration security best practices guidelines 
    if (questions_and_answers["Q12"].find("6") != -1):
        report.write("\n\n")

        phpconfig_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for PHP Configuration in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/PHP_Configuration_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for PHP Configuration \n\n")
        f.write(phpconfig_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/PHP_Configuration_Cheat_Sheet.md", "r").read())  
    
    # Ruby on Rails security best practices guidelines 
    if (questions_and_answers["Q12"].find("6") != -1):
        report.write("\n\n")

        rubirails_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Ruby on Rails in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Ruby_on_Rails_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Ruby on Rails \n\n")
        f.write(rubirails_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Ruby_on_Rails_Cheat_Sheet.md", "r").read()) 
    
    # Server Side Request Forgery Prevention security best practices guidelines 
    if (questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1):
        if (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            report.write("\n\n")

            ssrf_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for SSRF Prevention in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for SSRF Prevention \n\n")
            f.write(ssrf_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.md", "r").read())
    
    # Session Management security best practices guidelines
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            report.write("\n\n")

            session_management_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for Session Management in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/Session_Management_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for Session Management \n\n")
            f.write(session_management_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/Session_Management_Cheat_Sheet.md", "r").read())
    
    # Transport Layer Protection security best practices guidelines
    if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
        report.write("\n\n")

        tlp_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Transport Layer Protection in Markdown format \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Transport_Layer_Protection_Cheat_Sheet.md", "w")
        f.write("# Security Best Practices Guidelines for Transport Layer Protection \n\n")
        f.write(tlp_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Transport_Layer_Protection_Cheat_Sheet.md", "r").read())
     
    # Input Validation security best practices guidelines
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            if questions_and_answers["Q13"].find("1") != -1:
                report.write("\n\n")

                input_validation_sbpg = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Generate the security best practices guidelines for Input Validation in Markdown format \n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_best_practices_guidelines/Input_Validation_Cheat_Sheet.md", "w")
                f.write("# Security Best Practices Guidelines for Input Validation \n\n")
                f.write(input_validation_sbpg.choices[0]["text"].strip())
                f.close()

                report.write(open("security_best_practices_guidelines/Input_Validation_Cheat_Sheet.md", "r").read())
    
    # User Privacy Protection security best practices guidelines
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            report.write("\n\n")

            user_privacy_sbpg = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate the security best practices guidelines for User Privacy Protection in Markdown format. \n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_best_practices_guidelines/User_Privacy_Protection_Cheat_Sheet.md", "w")
            f.write("# Security Best Practices Guidelines for User Privacy Protection \n\n")
            f.write(user_privacy_sbpg.choices[0]["text"].strip())
            f.close()

            report.write(open("security_best_practices_guidelines/User_Privacy_Protection_Cheat_Sheet.md", "r").read())
         
    # Cryptography security best practices guidelines
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n\n")

        cryptography_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Cryptography in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/Cryptography_guide.md", "w")
        f.write("# Security Best Practices Guidelines for Cryptography \n\n")
        f.write(cryptography_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/Cryptography_guide.md", "r").read())

    # Secure Sofware Update security best practices guidelines
    if questions_and_answers["Q16"].find("1") != -1:
        report.write("\n\n")

        app_update_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Secure Update of Cloud-based Mobile Application in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/App_Update_guide.md", "w")
        f.write("# Security Best Practices Guidelines for Secure Application Update \n\n")
        f.write(app_update_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/App_Update_guide.md", "r").read())

    # Third-Party Application security best practices guidelines 
    if questions_and_answers["Q17"].find("1") != -1:
        report.write("\n\n")

        secure_third_party_sbpg = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate the security best practices guidelines for Secure Third-party Cloud-based Mobile Application in Markdown format. \n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_best_practices_guidelines/App_Third_Party_guide.md", "w")
        f.write("# Security Best Practices Guidelines for Secure Third-party Application \n\n")
        f.write(secure_third_party_sbpg.choices[0]["text"].strip())
        f.close()

        report.write(open("security_best_practices_guidelines/App_Third_Party_guide.md", "r").read())

    report.close()
    security_best_practices_convert_report()
    print("\n\n # Processing done! Check your security best practices guidelines in the GOOD_PRACTICES.pdf file")

"""
[Summary]: Method responsible for processing information about CMAME module
[Arguments]: No arguments
[Return]: No return
"""
def get_attack_models():
    print("")
    print("  Processing information.....")
    print("")

    report = open("ATTACKS_MAPPING.md", "w")
    report.write("# Final Attack Models Report  " + '\n')
    report.write("\n")

    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", "", "|", "", "|"))
    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", ":--------", "|", ":---------", "|"))

    '''
    for i in range( 0,len(table_for_report) ):
        report.write("| " + table_for_report[i][0] + " | " + table_for_report[i][1] + " | \n" )
    '''
    for i in range(0, len(table_for_report)):
        report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", table_for_report[i][0], "|", table_for_report[i][1], "|"))

    report.write("\n")

    # MitM attack model
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1):
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            mitm_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Man-in-th-Middle attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            mitm_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Man-in-th-Middle Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/mitm_attack_model.md", "w")
            f.write("# Man-in-th-Middle Attack \n\n")
            f.write(mitm_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/mitm_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Man-in-th-Middle Architectural Risk Analysis: \n\n")
            f.write(mitm_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/mitm_attack_model.md", "r").read())

            # Write de scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/mitmAttackTree.png)")
            report.write("\n\n")

    # Brute Force attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:

            brute_force_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Brute Force attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            brute_force_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Brute Force Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/brute_force_Attack_model.md", "w")
            f.write("# Brute Force Attack \n\n")
            f.write(brute_force_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/brute_force_Attack_model.md", "a")
            f.write("\n\n")
            f.write("## Brute Force Architectural Risk Analysis: \n\n")
            f.write(brute_force_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/brute_force_Attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/bruteForceAttackTree.png)")
            report.write("\n\n")

    # Eavesdropping attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:

            eavesdropping_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Eavesdropping attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            eavesdropping_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Eavesdropping Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/eavesdropping_attack_model.md", "w")
            f.write("# Eavesdropping Attack \n\n")
            f.write(eavesdropping_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/eavesdropping_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Eavesdropping Architectural Risk Analysis: \n\n")
            f.write(eavesdropping_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/eavesdropping_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/eavesdroppingAttackTree.png)")
            report.write("\n\n")

    # XSS attack model
    if questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1 and (questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1):
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            xss_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is XSS attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            xss_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of XSS Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/xss_attack_model.md", "w")
            f.write("# XSS Attack \n\n")
            f.write(xss_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/xss_attack_model.md", "a")
            f.write("\n\n")
            f.write("## XSS Architectural Risk Analysis: \n\n")
            f.write(xss_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/xss_attack_model.md", "r").read())

            # Write de scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/xssAttackTree.png)")
            report.write("\n\n")

    # CSRF attack model
    if questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1 and (questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1):
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            csrf_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is CSRF attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )
            print(csrf_definition.choices[0]["text"].strip())

            csrf_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of CSRF Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("csrf_attack_model.md", "w")
            f.write("# CSRF Attack \n\n")
            f.write(csrf_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/csrf_attack_model.md", "a")
            f.write("\n\n")
            f.write("## CSRF Architectural Risk Analysis: \n\n")
            f.write(csrf_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/csrf_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/csrfAttackTree.png)")
            report.write("\n\n")

    # Cookie Poisoning attack model
    if (questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1) and (questions_and_answers["Q18"].find("2") != -1 or questions_and_answers["Q18"].find("4") != -1):
        report.write("\n")
        report.write("\n")

        cookie_poisoning_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Cookie Poisoning attack? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        cookie_poisoning_risk_analysis = openai.Completion.create(
            model="text-davinci-003",
            prompt="Present the Architectural Risk Analysis of Cookie Poisoning Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("attack_models/cookie_poisoning_attack_model.md", "w")
        f.write("# Cookie Poisoning Attack \n\n")
        f.write(cookie_poisoning_definition.choices[0]["text"].strip())
        f.close()

        f = open("attack_models/cookie_poisoning_attack_model.md", "a")
        f.write("\n\n")
        f.write("## Cookie Poisoning Architectural Risk Analysis: \n\n")
        f.write(cookie_poisoning_risk_analysis.choices[0]["text"].strip())
        f.close()

        report.write(open("attack_models/cookie_poisoning_attack_model.md", "r").read())

        # Write de scheme in the report
        report.write("\n\n")
        report.write("![alt text](attack_models/cachePoisoningAttackTree.png)")
        report.write("\n\n")

    # Malicious QR Code attack model
    if (questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1) and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            malicious_qr_code_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Malicious QR Code attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            malicious_qr_code_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Malicious QR Code Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/malicious_qr_code_attack_model.md", "w")
            f.write("# Malicious QR Code Attack \n\n")
            f.write(malicious_qr_code_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/malicious_qr_code_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Malicious QR Code Architectural Risk Analysis: \n\n")
            f.write(malicious_qr_code_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/malicious_qr_code_attack_model.md", "r").read())

            # Write de scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/QRCodeAttackTree.png)")
            report.write("\n\n")

    # SQLi attack model
    if (questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1) and questions_and_answers["Q3"].find("1") != -1 :
        if questions_and_answers["Q5"].find("1") != -1 and questions_and_answers["Q6"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q7"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            sqli_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Malicious QR Code attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            sqli_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of SQLi Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/sqli_attack_model.md", "w")
            f.write("# SQLi Attack \n\n")
            f.write(sqli_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/sqli_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Malicious SQLi Architectural Risk Analysis: \n\n")
            f.write(sqli_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/sqli_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/sqliAttackTree.png)")
            report.write("\n\n")

    # Flooding attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:

            flooding_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Flooding attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            flooding_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Flooding Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/flooding_attack_model.md", "w")
            f.write("# Flooding Attack \n\n")
            f.write(flooding_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/flooding_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Flooding Architectural Risk Analysis: \n\n")
            f.write(flooding_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/flooding_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/floodingAttackTree.png)")
            report.write("\n\n")

    # Sniffing attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1):
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:

            sniffing_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Sniffing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            sniffing_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Sniffing Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/sniffing_attack_model.md", "w")
            f.write("# Sniffing Attack \n\n")
            f.write(sniffing_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/sniffing_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Sniffing Architectural Risk Analysis: \n\n")
            f.write(sniffing_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/sniffing_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/sniffingAttackTree.png)")
            report.write("\n\n")
    
    # Phishing attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q18"].find("2") != -1 or questions_and_answers["Q18"].find("4") != -1):
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            phishing_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Phishing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            phishing_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Phishing Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/phishing_attack_model.md", "w")
            f.write("# Phishing Attack \n\n")
            f.write(phishing_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/phishing_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Phishing Architectural Risk Analysis: \n\n")
            f.write(phishing_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/phishing_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/phishingAttackTree.png)")
            report.write("\n\n")

    # Botnet attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q18"].find("2") != -1 or questions_and_answers["Q18"].find("4") != -1):
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            botnet_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Botnet attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            botnet_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Botnet Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/botnet_attack_model.md", "w")
            f.write("# Botnet Attack \n\n")
            f.write(botnet_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/botnet_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Botnet Architectural Risk Analysis: \n\n")
            f.write(botnet_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/botnet_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/botnetAttackTree.png)")
            report.write("\n\n")

    # Session Hijacking attack model
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("2") != -1 and questions_and_answers["Q8"].find("3") != -1:
                report.write("\n")
                report.write("\n")

                session_hijacking_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Session Hijacking attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                session_hijacking_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Session Hijacking Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/session_hijacking_attack_model.md", "w")
                f.write("# Session Hijacking Attack \n\n")
                f.write(session_hijacking_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/session_hijacking_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Session Hijacking Architectural Risk Analysis: \n\n")
                f.write(session_hijacking_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/session_hijacking_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/sessionHijackingAttackTree.png)")
                report.write("\n\n")

    # Buffer Overflow attack model
    if questions_and_answers["Q1"].find("2") != -1 or questions_and_answers["Q1"].find("6") != -1 or questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q12"].find("2") != -1:
            report.write("\n")
            report.write("\n")

            buffer_overflow_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Buffer Overflow attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            buffer_overflow_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Buffer Overflow Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/buffer_overflow_attack_model.md", "w")
            f.write("# Buffer Overflow Attack \n\n")
            f.write(buffer_overflow_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/buffer_overflow_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Buffer Overflow Architectural Risk Analysis: \n\n")
            f.write(buffer_overflow_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/buffer_overflow_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/bufferOverflowAttackTree.png)")
            report.write("\n\n")

    # Spoofing attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 :
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            spoofing_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Spoofing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            spoofing_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Spoofing Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/spoofing_attack_model.md", "w")
            f.write("# Spoofing Attack \n\n")
            f.write(spoofing_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/spoofing_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Spoofing Architectural Risk Analysis: \n\n")
            f.write(spoofing_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/spoofing_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/spoofingAttackTree.png)")
            report.write("\n\n")

    # If the system was development for Android, iOS, Tizen and embedded platforms (Attack on VM at migration )
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            vm_migration_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is VM Migration attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            vm_migration_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of VM Migration Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/vm_migration_attack_model.md", "w")
            f.write("# VM Migration Attack \n\n")
            f.write(vm_migration_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/vm_migration_attack_model.md", "a")
            f.write("\n\n")
            f.write("## VM Migration Architectural Risk Analysis: \n\n")
            f.write(vm_migration_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/vm_migration_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/VMMigrationAttackTree.png)")
            report.write("\n\n")

    # Malicious Insider attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 :
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q22"].find("1") != -1:
                report.write("\n")
                report.write("\n")

                malicious_insider_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Malicious Insider attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                malicious_insider_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Malicious Insider Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/malicious_insider_attack_model.md", "w")
                f.write("# Malicious Insider Attack \n\n")
                f.write(malicious_insider_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/malicious_insider_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Malicious Insider Architectural Risk Analysis: \n\n")
                f.write(malicious_insider_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/malicious_insider_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/maliciousInsiderAttackTree.png)")
                report.write("\n\n")

    # VM Escape attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            vm_escape_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is VM Escape attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            vm_escape_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of VM Escape Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/vm_escape_attack_model.md", "w")
            f.write("# VM Escape Attack \n\n")
            f.write(vm_escape_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/vm_escape_attack_model.md", "a")
            f.write("\n\n")
            f.write("## VM Escape Architectural Risk Analysis: \n\n")
            f.write(vm_escape_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/vm_escape_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/vmEscapeAttackTree.png)")
            report.write("\n\n")

   # Side-Channel attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            side_channel_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Side-Channel attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            side_channel_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Side-Channel Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/side_channel_attack_model.md", "w")
            f.write("# Side-Channel Attack \n\n")
            f.write(side_channel_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/side_channel_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Side-Channel Architectural Risk Analysis: \n\n")
            f.write(side_channel_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/side_channel_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/sideChannelAttackTree.png)")
            report.write("\n\n")

    # Malware-as-a-Service attack model
        if (questions_and_answers["Q1"].find("1") != -1 or questions_and_answers["Q1"].find("2") != -1) or questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("5") != -1 or questions_and_answers["Q1"].find("6") != -1 or questions_and_answers["Q1"].find("7") != -1:
            if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
                report.write("\n")
                report.write("\n")

                maas_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Malware-as-a-Service attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                maas_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Malware-as-a-Service Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/malware_injection_attack_model.md", "w")
                f.write("# Malware-as-a-Service Attack \n\n")
                f.write(maas_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/malware_injection_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Malware-as-a-Service Architectural Risk Analysis: \n\n")
                f.write(maas_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/malware_injection_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/malwareInjectionAttackTree.png)")
                report.write("\n\n")

    # Tampering attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q22"].find("1") != -1:
                report.write("\n")
                report.write("\n")

                tampering_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Tampering attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                tampering_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Tampering Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/tampering_attack_model.md", "w")
                f.write("# Tampering Attack \n\n")
                f.write(tampering_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/tampering_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Tampering Architectural Risk Analysis: \n\n")
                f.write(tampering_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/tampering_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/tamperingAttackTree.png)")
                report.write("\n\n")
    
    # Bluejacking attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q21"].find("5") != -1:
                report.write("\n")
                report.write("\n")

                bluejacking_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Bluejacking attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                bluejacking_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Bluejacking Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/bluejacking_attack_model.md", "w")
                f.write("# Bluejacking Attack \n\n")
                f.write(bluejacking_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/bluejacking_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Bluejacking Architectural Risk Analysis: \n\n")
                f.write(bluejacking_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/bluejacking_attack_model.md", "r").read())

    # Bluesnarfing attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q21"].find("5") != -1:
                report.write("\n")
                report.write("\n")

                bluesnarfing_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Bluesnarfing attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                bluesnarfing_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Bluesnarfing Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/bluesnarfing_attack_model.md", "w")
                f.write("# Bluesnarfing Attack \n\n")
                f.write(bluesnarfing_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/bluesnarfing_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Bluesnarfing Architectural Risk Analysis: \n\n")
                f.write(bluesnarfing_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write("\n\n")
                report.write(open("attack_models/bluesnarfing_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/bluetoothAttackTree.png)")
                report.write("\n\n")

    # GPS Jamming attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q21"].find("7") != -1:
                report.write("\n")
                report.write("\n")

                gps_jamming_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is GPS Jamming attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                gps_jamming_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of GPS Jamming Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/gps_jamming_attack_model.md", "w")
                f.write("# GPS Jamming Attack \n\n")
                f.write(gps_jamming_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/gps_jamming_attack_model.md", "a")
                f.write("\n\n")
                f.write("## GPS Jamming Architectural Risk Analysis: \n\n")
                f.write(gps_jamming_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/gps_jamming_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/gps_jammingAttackTree.png)")
                report.write("\n\n")

    # Code injection attack model
    if questions_and_answers["Q1"].find("3") != -1 and questions_and_answers["Q1"].find("4") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
                report.write("\n")
                report.write("\n")

                code_injection_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Code Injection attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                code_injection_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Code Injection Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/code_injection_attack_model.md", "w")
                f.write("# Code Injection Attack \n\n")
                f.write(code_injection_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/code_injection_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Code Injection Architectural Risk Analysis: \n\n")
                f.write(code_injection_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/code_injection_attack_model.md", "r").read())


                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/codeInjectionAttackTree.png)")
                report.write("\n\n")

    # SSRF attack model
    if questions_and_answers["Q1"].find("3") != -1 and questions_and_answers["Q1"].find("4") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
                report.write("\n")
                report.write("\n")

                ssrf_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is SSRF attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                ssrf_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of SSRF Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/ssrf_attack_model.md", "w")
                f.write("# SSRF Attack \n\n")
                f.write(ssrf_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/ssrf_attack_model.md", "a")
                f.write("\n\n")
                f.write("## SSRF Vulnerabiligy Architectural Risk Analysis: \n\n")
                f.write(ssrf_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/ssrf_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/SSRFAttackTree.png)")
                report.write("\n\n")

    # Command Injection attack model
    if questions_and_answers["Q1"].find("3") != -1 and questions_and_answers["Q1"].find("4") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
                report.write("\n")
                report.write("\n")

                command_injection_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Command Injection attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                command_injection_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Command Injection Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/command_injection_attack_model.md", "w")
                f.write("# Command Injection Attack \n\n")
                f.write(command_injection_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/command_injection_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Command Injection Architectural Risk Analysis: \n\n")
                f.write(command_injection_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/command_injection_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/commandInjectionAttackTree.png)")
                report.write("\n\n")
    
    # Cellular Jamming attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q21"].find("1") != -1 or questions_and_answers["Q21"].find("2") != -1 or questions_and_answers["Q21"].find("3") != -1 or questions_and_answers["Q21"].find("4") != -1:
                report.write("\n")
                report.write("\n")

                cellular_jamming_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Cellular Jamming attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                cellular_jamming_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Cellular Jamming Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/cellular_jamming_attack_model.md", "w")
                f.write("# Cellular Jamming Attack \n\n")
                f.write(cellular_jamming_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/cellular_jamming_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Cellular Jamming Architectural Risk Analysis: \n\n")
                f.write(cellular_jamming_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/cellular_jamming_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/cellularJammingAttackTree.png)")
                report.write("\n\n")

    # Cryptanalysis attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            cryptanalysis_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Cryptanalysis attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            cryptanalysis_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Cryptanalysis Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/cryptanalysis_attack_model.md", "w")
            f.write("# Cryptanalysis Attack \n\n")
            f.write(cryptanalysis_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/cryptanalysis_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Cryptanalysis Architectural Risk Analysis: \n\n")
            f.write(cryptanalysis_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/cryptanalysis_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/cryptanalysisAttackTree.png)")
            report.write("\n\n")

    # Reverse Engineering attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            reverse_engineering_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Reverse Engineering attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            reverse_engineering_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Reverse Engineering Attack Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/reverse_engineering_attack_model.md", "w")
            f.write("# Reverse Engineering Attack \n\n")
            f.write(reverse_engineering_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/reverse_engineering_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Reverse Engineering Architectural Risk Analysis: \n\n")
            f.write(reverse_engineering_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write(open("attack_models/reverse_engineering_attack_model.md", "r").read())

            # Write the scheme in the report
            report.write("\n\n")
            report.write("![alt text](attack_models/reverseEngineeringAttackTree.png)")
            report.write("\n\n")

    # Audit Log Manipulation attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q14"].find("1") != -1:
                report.write("\n")
                report.write("\n")

                audit_log_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Audit Log Manipulation attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                audit_log_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Audit Log Manipulation Attack Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/audit_log_attack_model.md", "w")
                f.write("# Audit Log Manipulation Attack \n\n")
                f.write(audit_log_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/audit_log_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Audit Log Manipulation Architectural Risk Analysis: \n\n")
                f.write(audit_log_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/audit_log_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/auditLogManipulationAttackTree.png)")
                report.write("\n")
                report.write("\n")

    # Wi-Fi Jamming Attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q21"].find("6") != -1:
                report.write("\n")
                report.write("\n")

                wi_fi_jamming_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Wi-Fi Jamming attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                wi_fi_jamming_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Wi-Fi Jamming Attack Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/wi_fi_jamming_attack_model.md", "w")
                f.write("# Wi-Fi Jamming Attack \n\n")
                f.write(wi_fi_jamming_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/wi_fi_jamming_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Wi-Fi Jamming Architectural Risk Analysis: \n\n")
                f.write(wi_fi_jamming_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/wi_fi_jamming_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/Wi-Fi_JammingAttackTree.png)")
                report.write("\n\n")

    # Wi-Fi SSID Tracking attack model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q21"].find("6") != -1:
                report.write("\n")
                report.write("\n")

                wi_fi_ssid_tracking_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Wi-Fi SSID Tracking attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                wi_fi_ssid_tracking_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Wi-Fi SSID Tracking Attack Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/wi_fi_ssid_tracking_attack_model.md", "w")
                f.write("# Wi-Fi SSID Tracking Attack \n\n")
                f.write(wi_fi_ssid_tracking_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/wi_fi_ssid_tracking_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Wi-Fi SSID Tracking Architectural Risk Analysis: \n\n")
                f.write(wi_fi_ssid_tracking_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/wi_fi_ssid_tracking_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/Wi-Fi_TrackingAttackTree.png)")
                report.write("\n\n")

    # Byzantine Attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q21"].find("6") != -1:
                report.write("\n")
                report.write("\n")

                byzantine_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Byzantine attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                byzantine_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Byzantine Attack Vulnerability, according Common vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/byzantine_attack_model.md", "w")
                f.write("# Byzantine Attack \n\n")
                f.write(byzantine_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/byzantine_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Byzantine Architectural Risk Analysis: \n\n")
                f.write(byzantine_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write(open("attack_models/byzantine_attack_model.md", "r").read())

                # Write the scheme in the report
                report.write("\n\n")
                report.write("![alt text](attack_models/byzantineAttackTree.png)")
                report.write("\n\n")

    # Spectre Attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q22"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            spectre_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Spectre attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            spectre_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Spectre Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            spectre_attack_tree = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate a complex Spectre Attack Tree. Generate the response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/spectre_attack_model.md", "w")
            f.write("# Spectre Attack \n\n")
            f.write(spectre_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/spectre_attack_model.md.md", "a")
            f.write("\n\n")
            f.write("## Spectre Architectural Risk Analysis \n\n")
            f.write(spectre_risk_analysis.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/spectre_attack_model.md.md", "a")
            f.write("\n\n")
            f.write("## Spectre Attack Tree \n\n")
            f.write(spectre_attack_tree.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/spectre_attack_model.md", "r").read())
            report.write("\n\n")

    # Meltdown Attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q22"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            meltdown_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Meltdown attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            meltdown_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Meltdown Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            meltdown_attack_tree = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate a complex Meltdown Attack Tree. Generate the response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/meltdown_attack_model.md", "w")
            f.write("# Meltdown Attack \n\n")
            f.write(meltdown_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/meltdown_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Meltdown Architectural Risk Analysis: \n\n")
            f.write(meltdown_risk_analysis.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/meltdown_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Meltdown Attack Tree \n\n")
            f.write(meltdown_attack_tree.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/meltdown_attack_model.md", "r").read())
            report.write("\n\n")

    # Hardware Integrity Attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q22"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            hardware_integrity_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is the Hardware Integrity? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            hardware_integrity_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Hardware Integrity Vulnerability, according Common"
                       "Vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            hardware_integrity_attack_tree = openai.Completion.create(
                model="text-davinci-003",
                prompt="Generate a complex Hardware Integrity Attack Tree. Generate the response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/hardware_integrity_attack_model.md", "w")
            f.write("# Hardware Integrity Attack \n\n")
            f.write(hardware_integrity_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/hardware_integrity_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Hardware Integrity Architectural Risk Analysis \n\n")
            f.write(hardware_integrity_risk_analysis.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/hardware_integrity_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Hardware Integrity Attack Tree \n\n")
            f.write(hardware_integrity_attack_tree.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/hardware_integrity_attack_model.md", "r").read())
            report.write("\n\n")

    # Rowhammer Attack Model
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        rowhammer_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="What is Rowhammer? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        rowhammer_risk_analysis = openai.Completion.create(
            model="text-davinci-003",
            prompt="Present the Architectural Risk Analysis of Rowhammer Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        rowhammer_attack_tree = openai.Completion.create(
            model="text-davinci-003",
            prompt="Generate a complex Rowhammer Attack Tree. Generate the response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("attack_models/rowhammer_attack_model.md", "w")
        f.write("# Rowhammer Attack \n\n")
        f.write(rowhammer_definition.choices[0]["text"].strip())
        f.close()

        f = open("attack_models/rowhammer_attack_model.md", "a")
        f.write("\n\n")
        f.write("## Rowhammer Architectural Risk Analysis \n\n")
        f.write(rowhammer_risk_analysis.choices[0]["text"].strip())
        f.close()

        f = open("attack_models/rowhammer_attack_model.md", "a")
        f.write("\n\n")
        f.write("## Rowhammer Attack Tree \n\n")
        f.write(rowhammer_attack_tree.choices[0]["text"].strip())
        f.close()

        report.write("\n\n")
        report.write(open("attack_models/rowhammer_attack_model.md", "r").read())
        report.write("\n\n")

    # RF Interference on RFIDs attack Model
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            rf_interference_on_rfid_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is RF Interference on RFIDs? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            rf_interference_on_rfid_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of RF Interference on RFIDs Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/rf_interference_on_rfid_attack_model.md", "w")
            f.write("# RF Interference on RFIDs Attack \n\n")
            f.write(rf_interference_on_rfid_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/rf_interference_on_rfid_attack_model.md", "a")
            f.write("\n\n")
            f.write("## RF Interference On RFID Architectural Risk Analysis: \n\n")
            f.write(rf_interference_on_rfid_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/rf_interference_on_rfid_attack_model.md", "r").read())
            report.write("\n\n")

    # Node Tampering attack model
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q22"].find("1") != -1:
                report.write("\n")
                report.write("\n")

                node_tampering_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Node Tampering? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                node_tampering_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Node Tampering Attack Vulnerability, according Common "
                           "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/node_tampering_attack_model.md", "w")
                f.write("# Node Tampering Attack \n\n")
                f.write(node_tampering_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/node_tampering_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Node Tampering Architectural Risk Analysis: \n\n")
                f.write(node_tampering_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write("\n\n")
                report.write(open("attack_models/node_tampering_attack_model.md", "r").read())
                report.write("\n\n")

    # Node Jamming in WSNs attack model
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("10") != -1:
            report.write("\n")
            report.write("\n")

            node_jamming_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Node Jamming in WSNs? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            node_jamming_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Node Jamming in WSNs Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/node_jamming_attack_model.md", "w")
            f.write("# Node Jamming in WSNs Attack \n\n")
            f.write(node_jamming_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/node_jamming_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Node Jamming in WSNs Architectural Risk Analysis: \n\n")
            f.write(node_jamming_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/node_jamming_attack_model.md", "r").read())
            report.write("\n\n")

        # Sybil attack Model
        if questions_and_answers["Q1"].find("7") != -1:
            if questions_and_answers["Q21"].find("10") != -1:
                report.write("\n")
                report.write("\n")

                sybil_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What is Sybil attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                sybil_risk_analysis = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Present the Architectural Risk Analysis of Sybil Attack Vulnerability, according Common "
                           "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("attack_models/sybil_attack_model.md", "w")
                f.write("# Sybil Attack \n\n")
                f.write(sybil_definition.choices[0]["text"].strip())
                f.close()

                f = open("attack_models/sybil_attack_model.md", "a")
                f.write("\n\n")
                f.write("## Sybil Attack Architectural Risk Analysis: \n\n")
                f.write(sybil_risk_analysis.choices[0]["text"].strip())
                f.close()

                report.write("\n\n")
                report.write(open("attack_models/sybil_attack_model.md", "r").read())
                report.write("\n\n")

    # Malicious Node Injection attack model
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("10") != -1:
            report.write("\n")
            report.write("\n")

            malicious_node_injection_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is Malicious Node Injection attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            malicious_node_injection_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of Malicious Node Injection Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/malicious_node_injection_attack_model.md", "w")
            f.write("# Malicious Node Injection Attack \n\n")
            f.write(malicious_node_injection_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/malicious_node_injection_attack_model.md", "a")
            f.write("\n\n")
            f.write("## Malicious Node Injection Attack Architectural Risk Analysis: \n\n")
            f.write(malicious_node_injection_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/malicious_node_injection_attack_model.md", "r").read())
            report.write("\n\n")

    # RFID Spoofing attack model
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            rfid_Spoofing_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is RFID Spoofing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            rfid_Spoofing_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of RFID Spoofing Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/rfid_Spoofing_attack_model.md", "w")
            f.write("# RFID Spoofing Injection Attack \n\n")
            f.write(rfid_Spoofing_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/rfid_Spoofing_attack_model.md", "a")
            f.write("\n\n")
            f.write("## RFID Spoofing Attack Architectural Risk Analysis: \n\n")
            f.write(rfid_Spoofing_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/rfid_Spoofing_attack_model.md", "r").read())
            report.write("\n\n")

    # RFID Cloning attack model
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            rfid_cloning_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is RFID Cloning attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            rfid_cloning_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of RFID Cloning Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/rfid_cloning_attack_model.md", "w")
            f.write("# RFID Cloning Injection Attack \n\n")
            f.write(rfid_cloning_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/rfid_cloning_attack_model.md", "a")
            f.write("\n\n")
            f.write("## RFID Cloning Attack Architectural Risk Analysis: \n\n")
            f.write(rfid_cloning_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/rfid_cloning_attack_model.md", "r").read())
            report.write("\n\n")

    # RFID Unauthorized Access attack model
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            rfid_unauthorized_access_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="What is RFID Unauthorized Access attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            rfid_unauthorized_access_risk_analysis = openai.Completion.create(
                model="text-davinci-003",
                prompt="Present the Architectural Risk Analysis of RFID Unauthorized Access Attack Vulnerability, according Common "
                       "vulnerability Scoring System v3.1. Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("attack_models/rfid_unauthorized_access_attack_model.md", "w")
            f.write("# RFID Unauthorized Access Attack \n\n")
            f.write(rfid_unauthorized_access_definition.choices[0]["text"].strip())
            f.close()

            f = open("attack_models/rfid_unauthorized_access_attack_model.md", "a")
            f.write("\n\n")
            f.write("## RFID Unauthorized Access Attack Architectural Risk Analysis: \n\n")
            f.write(rfid_unauthorized_access_risk_analysis.choices[0]["text"].strip())
            f.close()

            report.write("\n\n")
            report.write(open("attack_models/rfid_unauthorized_access_attack_model.md", "r").read())
            report.write("\n\n")

    report.close()
    attack_models_convert_report()
    print("\n\n # Processing done! Check your attack models in the ATTACKS_MAPPING.pdf file")

"""
[Summary]: Method responsible for processing information about STSAT module
[Arguments]: No arguments
[Return]: No return
"""
def get_security_test_recommendation():
    print("")
    print("  Processing information.....")
    print("")

    report = open("TEST_SPECIFICATION.md", "w")
    report.write("# Final Security Test Specification and Tools Report  " + '\n')
    report.write("\n")

    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", "", "|", "", "|"))
    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", ":--------", "|", ":---------", "|"))

    '''
    for i in range( 0,len(table_for_report) ):
        report.write("| " + table_for_report[i][0] + " | " + table_for_report[i][1] + " | \n" )
    '''
    for i in range(0, len(table_for_report)):
        report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", table_for_report[i][0], "|", table_for_report[i][1], "|"))

    report.write("\n")


    # Testing DoS Jamming attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("1") != -1 or questions_and_answers["Q21"].find("2") != -1 or questions_and_answers["Q21"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            testing_dos_jamming = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing DoS or Cellular Jamming attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_dos_jamming_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the DoS or Cellular Jamming Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /dos_jamming_testing_guide.md", "w")
            f.write("# Testing the DoS or Cellular Jamming Attack \n\n")
            f.write(testing_dos_jamming.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /dos_jamming_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_dos_jamming_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /dos_jamming_testing_guide.md", "r").read())

    # Testing Wi-Fi Jamming attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("6") != -1:
            report.write("\n")
            report.write("\n")

            testing_wi_fi_jamming = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Wi-Fi Jamming attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_wi_fi_jamming_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Wi-Fi Jamming Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /wi_fi_jamming_testing_guide.md", "w")
            f.write("# Testing the Wi-Fi Jamming Attack \n\n")
            f.write(testing_wi_fi_jamming.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /wi_fi_jamming_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_wi_fi_jamming_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /wi_fi_jamming_testing_guide.md", "r").read())

    # Testing NFC Payment Replay attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("9") != -1:
            report.write("\n")
            report.write("\n")

            testing_nfc_payment_replay = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing NFC Payment Replay attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_nfc_payment_replay_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the NFC Payment Replay Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /nfc_payment_replay_testing_guide.md", "w")
            f.write("# Testing the NFC Payment Replay Attack \n\n")
            f.write(testing_nfc_payment_replay.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /nfc_payment_replay_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_nfc_payment_replay_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /nfc_payment_replay_testing_guide.md", "r").read())

    # Testing Orbital Jamming attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("1") != -1 or questions_and_answers["Q21"].find("2") != -1 or \
                questions_and_answers["Q21"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            testing_orbital_jamming = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Orbital Jamming attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_orbital_jamming_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Orbital Jamming Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /orbital_jamming_testing_guide.md", "w")
            f.write("# Testing the Orbital Jamming Attack \n\n")
            f.write(testing_orbital_jamming.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /orbital_jamming_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_orbital_jamming_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /orbital_jamming_testing_guide.md", "r").read())

    # Testing GPS Jamming attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("7") != -1:
            report.write("\n")
            report.write("\n")

            testing_gps_jamming = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing GPS Jamming attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_gps_jamming_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the GPS Jamming Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /gps_jamming_testing_guide.md", "w")
            f.write("# Testing the GPS Jamming Attack \n\n")
            f.write(testing_gps_jamming.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /gps_jamming_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_gps_jamming_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /gps_jamming_testing_guide.md", "r").read())

    # Testing Bluesnarfing attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("5") != -1:
            report.write("\n")
            report.write("\n")

            testing_bluesnarfing = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Bluesnarfing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_bluesnarfing_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Bluesnarfing Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /bluesnarfing_testing_guide.md", "w")
            f.write("# Testing the Bluesnarfing Attack \n\n")
            f.write(testing_bluesnarfing.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /bluesnarfing_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_bluesnarfing_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /bluesnarfing_testing_guide.md", "r").read())

    # Testing Bluejacking attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("5") != -1:
            report.write("\n")
            report.write("\n")

            testing_bluejacking = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Bluejacking attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_bluejacking_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Bluejacking Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /bluejacking_testing_guide.md", "w")
            f.write("# Testing the Bluejacking Attack \n\n")
            f.write(testing_bluejacking.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /bluejacking_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_bluejacking_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /bluejacking_testing_guide.md", "r").read())

    # Testing Wi-Fi SSID Tracking attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("6") != -1:
                report.write("\n")
                report.write("\n")

                testing_wi_fi_ssid_tracking = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="How to testing Wi-Fi SSID Tracking attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                tools_wi_fi_ssid_tracking_testing = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What tool to use for testing the Wi-Fi SSID Tracking? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_test_recommendation /wi_fi_ssid_tracking_testing_guide.md", "w")
                f.write("# Testing the Wi-Fi Jamming Attack \n\n")
                f.write(testing_wi_fi_ssid_tracking.choices[0]["text"].strip())
                f.close()

                f = open("security_test_recommendation /wi_fi_ssid_tracking_testing_guide.md", "a")
                f.write("\n\n")
                f.write("## Testing Tools: \n\n")
                f.write(tools_wi_fi_ssid_tracking_testing.choices[0]["text"].strip())
                f.close()

                report.write(open("security_test_recommendation /wi_fi_ssid_tracking_testing_guide.md", "r").read())

    # Testing Byzantine attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("6") != -1:
            report.write("\n")
            report.write("\n")

            testing_byzantine = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Byzantine attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_byzantine_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Byzantine Attacks? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /byzantine_testing_guide.md", "w")
            f.write("# Testing the Byzantin Attack \n\n")
            f.write(testing_byzantine.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /byzantine_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_byzantine_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /byzantine_testing_guide.md", "r").read())

    # Testing On-Off attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("10") != -1:
            report.write("\n")
            report.write("\n")

            testing_on_off = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing On-Off attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_on_off_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the On-Off Attacks? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /on_off_testing_guide.md", "w")
            f.write("# Testing the On-Off Attack \n\n")
            f.write(testing_on_off.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /on_off_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_on_off_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /on_off_testing_guide.md", "r").read())

    # Testing Malicious Insider attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1:
            report.write("\n")
            report.write("\n")

            testing_malicious_insider = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Malicious Insider attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_malicious_insider_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Malicious Insider Attacks? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /malicious_insider_testing_guide.md", "w")
            f.write("# Testing the Malicious Insider Attack \n\n")
            f.write(testing_malicious_insider.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /malicious_insider_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_malicious_insider_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /malicious_insider_testing_guide.md", "r").read())

    # Testing Sniffing attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1:
            report.write("\n")
            report.write("\n")

            testing_sniffing = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Sniffing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_sniffing_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Sniffing Attacks? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /sniffing_testing_guide.md", "w")
            f.write("# Testing the Sniffing Attack \n\n")
            f.write(testing_sniffing.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /sniffing_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_sniffing_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /sniffing_testing_guide.md", "r").read())

    # Testing MitM attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1:
            report.write("\n")
            report.write("\n")

            testing_mitm = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Man-in-the-Middle attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_mitm_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Man-in-the-Middle Attacks? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /mitm_testing_guide.md", "w")
            f.write("# Testing the Man-in-the-Middle Attack \n\n")
            f.write(testing_mitm.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /mitm_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_mitm_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /mitm_testing_guide.md", "r").read())

    # Testing Eavesdropping attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q18"].find("4") != -1 or questions_and_answers["Q18"].find("2") != -1:
            report.write("\n")
            report.write("\n")

            testing_eavesdropping = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Eavesdropping attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_eavesdropping_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Eavesdropping Attacks? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /eavesdropping_testing_guide.md", "w")
            f.write("# Testing the Eavesdropping Attack \n\n")
            f.write(testing_eavesdropping.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /eavesdropping_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_eavesdropping_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /eavesdropping_testing_guide.md", "r").read())

    # Testing CSRF attack
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_csrf = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing CSRF attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_csrf_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the CSRF Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /csrf_testing_guide.md", "w")
            f.write("# Testing the CSRF Attack \n\n")
            f.write(testing_csrf.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /csrf_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_csrf_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /csrf_testing_guide.md", "r").read())

    # Testing SQLi attacks
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_sqli = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing SQLi attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_sqli_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the SQLi Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /sqli_testing_guide.md", "w")
            f.write("# Testing the SQLi Attack \n\n")
            f.write(testing_sqli.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /sqli_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_sqli_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /sqli_testing_guide.md", "r").read())

    # Testing XSS attacks
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_xss = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing XSS attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            testing_xss_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the XSS Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /xss_testing_guide.md", "w")
            f.write("# Testing the XSS Attack \n\n")
            f.write(testing_xss.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /xss_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(testing_xss_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /xss_testing_guide.md", "r").read())

    # Testing SSRF attacks
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_ssrf = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing SSRF attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_ssrf_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the SSRF Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /ssrf_testing_guide.md", "w")
            f.write("# Testing the SSRF Attack \n\n")
            f.write(testing_ssrf.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /ssrf_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_ssrf_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /ssrf_testing_guide.md", "r").read())

    # Testing Command Injection attacks
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_command_injection = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Command Injection attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_command_injection_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Command Injection Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /command_injection_testing_guide.md", "w")
            f.write("# Testing the Command Injection Attack \n\n")
            f.write(testing_command_injection.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /command_injection_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_command_injection_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /command_injection_testing_guide.md", "r").read())

    # Testing Command Injection attacks
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_code_injection = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Code Injection attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_code_injection_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Code Injection Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /code_injection_testing_guide.md", "w")
            f.write("# Testing the Code Injection Attack \n\n")
            f.write(testing_code_injection.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /code_injection_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_code_injection_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /code_injection_testing_guide.md", "r").read())

    # Testing Phishing attacks
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_phishing = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Phishing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_phishing_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Phishing Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /phishing_testing_guide.md", "w")
            f.write("# Testing the Phishing Attack \n\n")
            f.write(testing_phishing.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /phishing_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_phishing_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /phishing_testing_guide.md", "r").read())

    # Testing Pharming attack
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_pharming = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Pharming attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_pharming_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Pharming Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /pharming_testing_guide.md", "w")
            f.write("# Testing the Pharming Attack \n\n")
            f.write(testing_pharming.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /pharming_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_pharming_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /pharming_testing_guide.md", "r").read())

    # Testing Spoofing attack
    if questions_and_answers["Q1"].find("4") != -1 or questions_and_answers["Q1"].find("3") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and \
                questions_and_answers["Q6"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            testing_spoofing = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Spoofing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_spoofing_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Spoofing Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /spoofing_testing_guide.md", "w")
            f.write("# Testing the Spoofing Attack \n\n")
            f.write(testing_spoofing.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /spoofing_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_spoofing_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /spoofing_testing_guide.md", "r").read())

    # Testing Session Fixation attack
    if questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("1") != -1 and questions_and_answers["Q8"].find("2") != -1 and questions_and_answers["Q8"].find("3") != -1:
                if questions_and_answers["Q18"].find("2") != -1 or questions_and_answers["Q18"].find("4") != -1:
                    report.write("\n")
                    report.write("\n")

                    testing_session_fixation = openai.Completion.create(
                        model="text-davinci-003",
                        prompt="How to testing Session Fixation attack? Generate response in Markdown format.\n\n",
                        max_tokens=500,
                        n=1,
                        best_of=2,
                    )

                    tools_session_fixation_testing = openai.Completion.create(
                        model="text-davinci-003",
                        prompt="What tool to use for testing the Session Fixation Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                        max_tokens=500,
                        n=1,
                        best_of=2,
                    )

                    f = open("security_test_recommendation /session_fixation_testing_guide.md", "w")
                    f.write("# Testing the Session Fixation Attack \n\n")
                    f.write(testing_session_fixation.choices[0]["text"].strip())
                    f.close()

                    f = open("security_test_recommendation /session_fixation_testing_guide.md", "a")
                    f.write("\n\n")
                    f.write("## Testing Tools: \n\n")
                    f.write(tools_session_fixation_testing.choices[0]["text"].strip())
                    f.close()

                    report.write(open("security_test_recommendation /session_fixation_testing_guide.md", "r").read())

    # Testing Session Hijacking attacks
    if questions_and_answers["Q1"].find("3") != -1 or questions_and_answers["Q1"].find("4") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("1") != -1 and questions_and_answers["Q8"].find("2") != -1 and questions_and_answers["Q8"].find("3") != -1:
                if questions_and_answers["Q18"].find("2") != -1 or questions_and_answers["Q18"].find("4") != -1:
                    report.write("\n")
                    report.write("\n")

                    testing_session_hijacking = openai.Completion.create(
                        model="text-davinci-003",
                        prompt="How to testing Session Hijacking attack? Generate response in Markdown format.\n\n",
                        max_tokens=500,
                        n=1,
                        best_of=2,
                    )

                    tools_session_hijacking_testing = openai.Completion.create(
                        model="text-davinci-003",
                        prompt="What tool to use for testing the Session Hijacking Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                        max_tokens=500,
                        n=1,
                        best_of=2,
                    )

                    f = open("security_test_recommendation /session_hijacking_testing_guide.md", "w")
                    f.write("# Testing the Session Hijacking Attack \n\n")
                    f.write(testing_session_hijacking.choices[0]["text"].strip())
                    f.close()

                    f = open("security_test_recommendation /session_hijacking_testing_guide.md", "a")
                    f.write("\n\n")
                    f.write("## Testing Tools: \n\n")
                    f.write(tools_session_hijacking_testing.choices[0]["text"].strip())
                    f.close()

                    report.write(open("security_test_recommendation /session_hijacking_testing_guide.md", "r").read())

    # Testing Access Point Hijacking attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("6") != -1:
            report.write("\n")
            report.write("\n")

            testing_access_point_hijacking = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Access Point Hijacking attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_access_point_hijacking_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Access Point Hijacking Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /access_point_hijacking_testing_guide.md", "w")
            f.write("# Testing the Access Point Hijacking Attack \n\n")
            f.write(testing_access_point_hijacking.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /access_point_hijacking_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_access_point_hijacking_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /access_point_hijacking_testing_guide.md", "r").read())

    # Testing Cellular Rogue Base Station attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q21"].find("1") != -1 or questions_and_answers["Q21"].find("2") != -1 or \
                questions_and_answers["Q21"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            testing_crbs = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Cellular Rogue Base Station attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_crbs_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Cellular Rogue Base Station Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /cellular_rogue_base_station_testing_guide.md", "w")
            f.write("# Testing the Cellular Rogue Base Station Attack \n\n")
            f.write(testing_crbs.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /cellular_rogue_base_station_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_crbs_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /cellular_rogue_base_station_testing_guide.md", "r").read())

    # Testing GPS Spoofing attack
    if questions_and_answers["Q21"].find("7") != -1:
        report.write("\n")
        report.write("\n")

        testing_gps_spoofing = openai.Completion.create(
            model="text-davinci-003",
            prompt="How to testing GPS Spoofing? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        tools_gps_spoofing_testing = openai.Completion.create(
            model="text-davinci-003",
            prompt="What tool to use for testing the GPS Spoofing Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_test_recommendation /gps_spoofing_testing_guide.md", "w")
        f.write("# Testing the GPS Spoofing Attack \n\n")
        f.write(testing_gps_spoofing.choices[0]["text"].strip())
        f.close()

        f = open("security_test_recommendation /gps_spoofing_testing_guide.md", "a")
        f.write("\n\n")
        f.write("## Testing Tools: \n\n")
        f.write(tools_gps_spoofing_testing.choices[0]["text"].strip())
        f.close()

        report.write(open("security_test_recommendation /gps_spoofing_testing_guide.md", "r").read())

    # Testing RF Interference on RFIDs attack
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            testing_rf_interference_on_rfid = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing RF Interference on RFIDs? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_rf_interference_on_rfid_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the RF Interference on RFIDs Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /rf_interference_on_rfid_testing_guide.md", "w")
            f.write("# Testing the RF Interference on RFIDs Attack \n\n")
            f.write(testing_rf_interference_on_rfid.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /rf_interference_on_rfid_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_rf_interference_on_rfid_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /rf_interference_on_rfid_testing_guide.md", "r").read())

    # Testing Node Tampering attack
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q22"].find("1") != -1:
                report.write("\n")
                report.write("\n")

                testing_node_tampering = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="How to testing Node Tampering attack? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                tools_node_tampering_testing = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What tool to use for testing the Node Tampering Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_test_recommendation /node_tampering_testing_guide.md", "w")
                f.write("# Testing the Node Tampering Attack \n\n")
                f.write(testing_node_tampering.choices[0]["text"].strip())
                f.close()

                f = open("security_test_recommendation /node_tampering_testing_guide.md", "a")
                f.write("\n\n")
                f.write("## Testing Tools: \n\n")
                f.write(tools_node_tampering_testing.choices[0]["text"].strip())
                f.close()

                report.write(open("security_test_recommendation /node_tampering_testing_guide.md", "r").read())

    # Testing Node Jamming in WSNs attack
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("10") != -1:
            report.write("\n")
            report.write("\n")

            testing_node_jamming = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Node Jamming? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_node_jamming_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Node Jamming Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /node_jamming_testing_guide.md", "w")
            f.write("# Testing the Node Jamming Attack \n\n")
            f.write(testing_node_jamming.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /node_jamming_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_node_jamming_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /node_jamming_testing_guide.md", "r").read())

        # Testing Sybil attack
        if questions_and_answers["Q1"].find("7") != -1:
            if questions_and_answers["Q21"].find("10") != -1:
                report.write("\n")
                report.write("\n")

                testing_sybil = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="How to testing Sybil? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                tools_sybil_testing = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What tool to use for testing the Sybil Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_test_recommendation /sybil_testing_guide.md", "w")
                f.write("# Testing the Sybil Attack \n\n")
                f.write(testing_sybil.choices[0]["text"].strip())
                f.close()

                f = open("security_test_recommendation /sybil_testing_guide.md", "a")
                f.write("\n\n")
                f.write("## Testing Tools: \n\n")
                f.write(tools_sybil_testing.choices[0]["text"].strip())
                f.close()

                report.write(open("security_test_recommendation /sybil_testing_guide.md", "r").read())

    # Testing Malicious Node Injection attack
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("10") != -1:
            report.write("\n")
            report.write("\n")

            malicious_node_injection_spoofing = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Malicious Node Injection? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_malicious_node_injection_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Malicious Node Injection Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /malicious_node_injection_testing_guide.md", "w")
            f.write("# Testing the Malicious Node Injection Attack \n\n")
            f.write(malicious_node_injection_spoofing.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /malicious_node_injection_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_malicious_node_injection_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /malicious_node_injection_testing_guide.md", "r").read())

    # Testing RFID Spoofing attack
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            testing_rfid_Spoofing = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing RFID Spoofing attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_rfid_Spoofing_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the RFID Spoofing Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /rfid_Spoofing_testing_guide.md", "w")
            f.write("# Testing the RFID Spoofing Attack \n\n")
            f.write(testing_rfid_Spoofing.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /rfid_Spoofing_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_rfid_Spoofing_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /rfid_Spoofing_testing_guide.md", "r").read())

    # Testing RFID Cloning attack
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            testing_rfid_cloning = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing RFID Cloning? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_rfid_cloning_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the RFID Cloning Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /rfid_cloning_testing_guide.md", "w")
            f.write("# Testing the RFID Cloning Attack \n\n")
            f.write(testing_rfid_cloning.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /rfid_cloning_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_rfid_cloning_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /rfid_cloning_testing_guide.md", "r").read())

    # Testing RFID Unauthorized Access attack
    if questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q21"].find("8") != -1:
            report.write("\n")
            report.write("\n")

            testing_rfid_unauthorized_access = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing RFID Unauthorized Access attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_rfid_unauthorized_access_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the RFID Unauthorized Access Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /rfid_unauthorized_access_testing_guide.md", "w")
            f.write("# Testing the RFID Unauthorized Access Attack \n\n")
            f.write(testing_rfid_unauthorized_access.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /rfid_unauthorized_access_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(testing_rfid_unauthorized_access.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /rfid_unauthorized_access_testing_guide.md", "r").read())

    # Testing Botnet attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            testing_botnet = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Botnet attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_botnet_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Botnet Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /botnet_testing_guide.md", "w")
            f.write("# Testing the Botnet Attack \n\n")
            f.write(testing_botnet.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /botnet_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_botnet_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /botnet_testing_guide.md", "r").read())

    # Testing Malware-as-a-Service attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            testing_maas = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Malware-as-a-Service attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_maas_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Malware-as-a-Service Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /maas_testing_guide.md", "w")
            f.write("# Testing the Malware-as-a-Service Attack \n\n")
            f.write(testing_maas.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /maas_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_maas_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /maas_testing_guide.md", "r").read())

    # Testing Buffer Overflow attacks
    if questions_and_answers["Q1"].find("2") != -1 or questions_and_answers["Q1"].find("6") != -1 or questions_and_answers["Q1"].find("7") != -1:
        if questions_and_answers["Q12"].find("2") != -1:
            report.write("\n")
            report.write("\n")
            testing_buffer_overflow = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Buffer Overflow? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_buffer_overflow_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Buffer Overflow Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /buffer_overflow_testing_guide.md", "w")
            f.write("# Testing the Buffer Overflow Attack \n\n")
            f.write(testing_buffer_overflow.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /buffer_overflow_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_buffer_overflow_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /buffer_overflow_testing_guide.md", "r").read())

    # Testing Bypassing Physical Security attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            testing_bps = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Bypassing Physical Security attack? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_bps_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Bypassing Physical Security Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /bps_testing_guide.md", "w")
            f.write("# Testing the Bypassing Physical Security Attack \n\n")
            f.write(testing_bps.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /bps_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_bps_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /bps_testing_guide.md", "r").read())

    # Testing Physical theft attack
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            testing_physical_theft = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Physical Theft? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_physical_theft_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Physical Theft Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /physical_theft_testing_guide.md", "w")
            f.write("# Testing the Physical Theft Attack \n\n")
            f.write(testing_physical_theft.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /physical_theft_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_physical_theft_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /physical_theft_testing_guide.md", "r").read())

        # Testing VM Migration attacks
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or \
                    questions_and_answers["Q8"].find("3") != -1:
                report.write("\n")
                report.write("\n")

                testing_vm_migration = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="How to testing VM Migration? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                tools_vm_migration_testing = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What tool to use for testing the VM Migration Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_test_recommendation /vm_migration_testing_guide.md", "w")
                f.write("# Testing the VM Migration Attack \n\n")
                f.write(testing_vm_migration.choices[0]["text"].strip())
                f.close()

                f = open("security_test_recommendation /vm_migration_testing_guide.md", "a")
                f.write("\n\n")
                f.write("## Testing Tools: \n\n")
                f.write(tools_vm_migration_testing.choices[0]["text"].strip())
                f.close()

                report.write(open("security_test_recommendation /vm_migration_testing_guide.md", "r").read())

    # Testing Side-Channel attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q22"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            testing_side_channel = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Side-Channel? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_side_channel_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Side-Channel Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /side_channel_testing_guide.md", "w")
            f.write("# Testing the Side-Channel Attack \n\n")
            f.write(testing_side_channel.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /side_channel_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_side_channel_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /side_channel_testing_guide.md", "r").read())

    # Testing Spectre attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q22"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            testing_spectre = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Spectre? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_spectre_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Spectre Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /spectre_testing_guide.md", "w")
            f.write("# Testing the Spectre Attack \n\n")
            f.write(testing_spectre.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /spectre_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_spectre_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /spectre_testing_guide.md", "r").read())

    # Testing Meltdown attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
            questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q22"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            testing_meltdown = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Meltdown? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_meltdown_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Meltdown Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /meltdown_testing_guide.md", "w")
            f.write("# Testing the Meltdown Attack \n\n")
            f.write(testing_meltdown.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /meltdown_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_meltdown_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /meltdown_testing_guide.md", "r").read())

    # Testing Hardware Integrity attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q22"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            testing_hardware_integrity = openai.Completion.create(
                model="text-davinci-003",
                prompt="How to testing Hardware Integrity? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            tools_hardware_integrity_testing = openai.Completion.create(
                model="text-davinci-003",
                prompt="What tool to use for testing the Hardware Integrity Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_test_recommendation /hardware_integrity_testing_guide.md", "w")
            f.write("# Testing Hardware Integrity Attack \n\n")
            f.write(testing_hardware_integrity.choices[0]["text"].strip())
            f.close()

            f = open("security_test_recommendation /hardware_integrity_testing_guide.md", "a")
            f.write("\n\n")
            f.write("## Testing Tools: \n\n")
            f.write(tools_hardware_integrity_testing.choices[0]["text"].strip())
            f.close()

            report.write(open("security_test_recommendation /hardware_integrity_testing_guide.md", "r").read())

    # Testing Rowhammer attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
                questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        testing_rowhammer = openai.Completion.create(
            model="text-davinci-003",
            prompt="How to testing Rowhammer? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        tools_rowhammer_testing = openai.Completion.create(
            model="text-davinci-003",
            prompt="What tool to use for testing the Rowhammer Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_test_recommendation /rowhammer_testing_guide.md", "w")
        f.write("# Testing Rowhammer Attack \n\n")
        f.write(testing_rowhammer.choices[0]["text"].strip())
        f.close()

        f = open("security_test_recommendation /rowhammer_testing_guide.md", "a")
        f.write("\n\n")
        f.write("## Testing Tools: \n\n")
        f.write(tools_rowhammer_testing.choices[0]["text"].strip())
        f.close()

        report.write(open("security_test_recommendation /rowhammer_testing_guide.md", "r").read())

    # Testing Reverse Engineering attacks
    if questions_and_answers["Q1"].find("1") != -1 and questions_and_answers["Q5"].find("2") != -1:
        if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1:
            if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
                report.write("\n")
                report.write("\n")

                testing_rea_testing = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="How to testing Side-Channel? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                tools_rea_testing_testing = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="What tool to use for testing the Side-Channel Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_test_recommendation /rea_testing_guide.md", "w")
                f.write("# Testing the Side-Channel Attack \n\n")
                f.write(testing_rea_testing.choices[0]["text"].strip())
                f.close()

                f = open("security_test_recommendation /rea_testing_guide.md", "a")
                f.write("\n\n")
                f.write("## Testing Tools: \n\n")
                f.write(tools_rea_testing_testing.choices[0]["text"].strip())
                f.close()

                report.write(open("security_test_recommendation /rea_testing_guide.md", "r").read())

    # Testing VM Escape attacks
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        testing_vm_escape_testing = openai.Completion.create(
            model="text-davinci-003",
            prompt="How to testing VM Escape? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        tools_vm_escape_testing_testing = openai.Completion.create(
            model="text-davinci-003",
            prompt="What tool to use for testing the VM Escape Attack? Generate response in a table by Target Testing, Testing Technique (White-box, Grey-box, Black-box), Test Analysis (Dynamic, Static, Hybrid), Test Method, Test Tool, Mobile Plataform, in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_test_recommendation /vm_escape_testing_guide.md", "w")
        f.write("# Testing the VM Escape Attack \n\n")
        f.write(testing_vm_escape_testing.choices[0]["text"].strip())
        f.close()

        f = open("security_test_recommendation /vm_escape_testing_guide.md", "a")
        f.write("\n\n")
        f.write("## Testing Tools: \n\n")
        f.write(tools_vm_escape_testing_testing.choices[0]["text"].strip())
        f.close()

        report.write(open("security_test_recommendation /vm_escape_testing_guide.md", "r").read())
    
    report.close()
    security_test_recommendation_convert_report()
    print("\n\n # Processing done! Check your security test specification and automation tools in the TEST_SPECIFICATION.pdf file")

"""
[Summary]: Method responsible for processing information about SME module
[Arguments]: No arguments
[Return]: No return
"""
def get_mechanisms():
    print("")
    print("  Processing information.....")
    print("")

    report = open("SECURITY_MECHANISMS.md", "w")
    report.write("# Final Security Mechanisms Report " + '\n')
    report.write("\n")

    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", "", "|", "", "|"))
    report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", ":--------", "|", ":---------", "|"))

    '''
    for i in range( 0,len(table_for_report) ):
        report.write("| " + table_for_report[i][0] + " | " + table_for_report[i][1] + " | \n" )
    '''
    for i in range(0, len(table_for_report)):
        report.write("{:3}{:25}{:3}{:60}{:3}\n".format("|", table_for_report[i][0], "|", table_for_report[i][1], "|"))

    report.write("\n")

    # Backup mechanisms
    if questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        backup_mechanism_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Security Backup Mechanisms in cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        backup_mechanism_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of secure backup mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/backup_mechanism_guide.md", "w")
        f.write("# Security Backup Mechanisms \n\n")
        f.write(backup_mechanism_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/backup_mechanism_guide.md", "a")
        f.write("\n\n")
        f.write("## Backup Mechanisms Examples: \n\n")
        f.write(backup_mechanism_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/backup_mechanism_guide.md", "r").read())

    # Audit mechanisms
    if questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        audit_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Security Audit Mechanisms in cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        audit_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Security Audit Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/audit_mechanisms_guide.md", "w")
        f.write("# Security Audit Mechanisms \n\n")
        f.write(audit_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/audit_mechanisms_guide.md", "a")
        f.write("\n\n")
        f.write("## Audit Mechanisms Examples: \n\n")
        f.write(audit_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/audit_mechanisms_guide.md", "r").read())

    # Cryptographic Algorithms Mechanisms
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        report.write("\n")
        report.write("\n")

        cryptographic_algorithms_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Cryptographic Algorithms Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        cryptographic_algorithms_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure Cryptographic Algorithms Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/cryptographic_algorithms_mechanisms.md", "w")
        f.write("# Cryptographic Algorithms Mechanisms \n\n")
        f.write(cryptographic_algorithms_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/cryptographic_algorithms_mechanisms.md", "a")
        f.write("\n\n")
        f.write("## Cryptographic Algorithms Mechanisms Examples: \n\n")
        f.write(cryptographic_algorithms_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/cryptographic_algorithms_mechanisms.md", "r").read())

    # Biometric Authentication Mechanisms
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q4"].find("1") != -1:

        biometric_authentication_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Biometric Authentication Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        biometric_authentication_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure Biometric Authentication Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/biometric_authentication_mechanism.md", "w")
        f.write("# Biometric Authentication Mechanisms \n\n")
        f.write(biometric_authentication_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/biometric_authentication_mechanism.md", "a")
        f.write("\n\n")
        f.write("## Biometric Authentication Mechanisms Examples: \n\n")
        f.write(biometric_authentication_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/biometric_authentication_mechanism.md", "r").read())

    # Channel-based Authentication Mechanisms
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q4"].find("2") != -1:

        channel_based_authentication_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Channel-based Authentication Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        channel_based_authentication_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure Channel-based Authentication Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/channel_based_authentication_mechanism.md", "w")
        f.write("# Channel-based Authentication Mechanisms \n\n")
        f.write(channel_based_authentication_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/channel_based_authentication_mechanism.md", "a")
        f.write("\n\n")
        f.write("## Channel-based Authentication Mechanisms Examples: \n\n")
        f.write(channel_based_authentication_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/channel_based_authentication_mechanism.md", "r").read())

    # Factors-based Authentication Mechanisms
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q4"].find("3") != -1:

        factors_based_authentication_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Factors-based Authentication Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        factors_based_authentication_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure Factors-based Authentication Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/factors_based_authentication_mechanism.md", "w")
        f.write("# Factors-based Authentication Mechanisms \n\n")
        f.write(factors_based_authentication_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/factors_based_authentication_mechanism.md", "a")
        f.write("\n\n")
        f.write("## Factors-based Authentication Mechanisms Examples: \n\n")
        f.write(factors_based_authentication_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/factors_based_authentication_mechanism.md", "r").read())
        
    # ID-based Authentication Mechanisms
    if questions_and_answers["Q3"].find("1") != -1 and questions_and_answers["Q4"].find("4") != -1:

        id_based_authentication_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of ID-based Authentication Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        id_based_authentication_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure ID-based Authentication Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/id_based_authentication_mechanism.md", "w")
        f.write("# ID-based Authentication Mechanisms \n\n")
        f.write(id_based_authentication_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/id_based_authentication_mechanism.md", "a")
        f.write("\n\n")
        f.write("## ID-based Authentication Mechanisms Examples: \n\n")
        f.write(id_based_authentication_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/id_based_authentication_mechanism.md", "r").read())

    # Cryptographic Protocolos Mechanisms
    if questions_and_answers["Q5"].find("1") != -1 and (
            questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
            report.write("\n")
            report.write("\n")

            cryptographic_protocols_mechanisms_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="Definition of Cryptographic Protocols Mechanisms in Cloud-based mobile apps.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            cryptographic_protocols_mechanisms_examples = openai.Completion.create(
                model="text-davinci-003",
                prompt="Enumerate examples of Secure Cryptographic Protocols Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_mechanisms/cryptographic_protocols_mechanisms.md", "w")
            f.write("# Cryptographic Protocols Authentication Mechanisms \n\n")
            f.write(cryptographic_protocols_mechanisms_definition.choices[0]["text"].strip())
            f.close()

            f = open("security_mechanisms/cryptographic_protocols_mechanisms.md", "a")
            f.write("\n\n")
            f.write("## Cryptographic Protocols Mechanisms Examples: \n\n")
            f.write(cryptographic_protocols_mechanisms_examples.choices[0]["text"].strip())
            f.close()

            report.write(open("security_mechanisms/cryptographic_protocols_mechanisms.md", "r").read())

    # Access Control Mechanisms
    if questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            access_control_mechanisms_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="Definition of Security Access Control Mechanisms in Cloud-based mobile apps.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            access_control_mechanisms_examples = openai.Completion.create(
                model="text-davinci-003",
                prompt="Enumerate examples of Secure Access Control Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_mechanisms/access_control_mechanisms.md", "w")
            f.write("# Access Control Mechanisms \n\n")
            f.write(access_control_mechanisms_definition.choices[0]["text"].strip())
            f.close()

            f = open("security_mechanisms/access_control_mechanisms.md", "a")
            f.write("\n\n")
            f.write("## Access Control Mechanisms Examples: \n\n")
            f.write(access_control_mechanisms_examples.choices[0]["text"].strip())
            f.close()

            report.write(open("security_mechanisms/access_control_mechanisms.md", "r").read())

    # Inspection Mechanisms
    if questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            report.write("\n")
            report.write("\n")

            device_inspection_mechanisms_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="Definition of Inspection Mechanisms in Cloud-based mobile apps.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            device_inspection_mechanisms_examples = openai.Completion.create(
                model="text-davinci-003",
                prompt="Enumerate examples of Inspection Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_mechanisms/inspection_mechanisms.md", "w")
            f.write("# Inspection Mechanisms \n\n")
            f.write(device_inspection_mechanisms_definition.choices[0]["text"].strip())
            f.close()

            f = open("security_mechanisms/inspection_mechanisms.md", "a")
            f.write("\n\n")
            f.write("## Inspection Mechanisms Examples: \n\n")
            f.write(device_inspection_mechanisms_examples.choices[0]["text"].strip())
            f.close()

            report.write(open("security_mechanisms/inspection_mechanisms.md", "r").read())

    # Logging Mechanisms
    if questions_and_answers["Q5"].find("1") != -1:
        if questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1:
            if questions_and_answers["Q15"].find("1") != -1:
                report.write("\n")
                report.write("\n")

                device_logging_mechanisms_definition = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Definition of Logging Mechanisms in Cloud-based mobile apps.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )
                device_inspection_mechanisms_examples = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Enumerate examples of Logging Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
                    max_tokens=500,
                    n=1,
                    best_of=2,
                )

                f = open("security_mechanisms/logging_mechanisms.md", "w")
                f.write("# Logging Mechanisms \n\n")
                f.write(device_inspection_mechanisms_definition.choices[0]["text"].strip())
                f.close()

                f = open("security_mechanisms/logging_mechanisms.md", "a")
                f.write("\n\n")
                f.write("## Logging Mechanisms Examples: \n\n")
                f.write(device_inspection_mechanisms_examples.choices[0]["text"].strip())
                f.close()

                report.write(open("security_mechanisms/logging_mechanisms.md", "r").read())

    # Device Tamper Detection Mechanisms
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1) and questions_and_answers["Q22"].find(
            "1") != -1:
        report.write("\n")
        report.write("\n")

        device_tamper_detection_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Security Device Tamper Detection Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        device_tamper_detection_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure Device Tamper Detection Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/device_tamper_detection_mechanisms.md", "w")
        f.write("# Device Detection Mechanisms \n\n")
        f.write(device_tamper_detection_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/device_tamper_detection_mechanisms.md", "a")
        f.write("\n\n")
        f.write("## Device Tamper Detection Mechanisms Examples: \n\n")
        f.write(device_tamper_detection_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/device_tamper_detection_mechanisms.md", "r").read())

    # Physical Location Mechanisms
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("1") != -1 or questions_and_answers["Q8"].find("2") != -1 or
                questions_and_answers["Q8"].find("3") != -1) and questions_and_answers["Q22"].find("1") != -1:
        report.write("\n")
        report.write("\n")

        physical_location_mechanisms_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Security Physical Location Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        physical_location_mechanisms_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure Physical Location Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/physical_location_mechanism.md", "w")
        f.write("# Physical Location Mechanisms \n\n")
        f.write(physical_location_mechanisms_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/physical_location_mechanism.md", "a")
        f.write("\n\n")
        f.write("## Physical Location Mechanisms Examples: \n\n")
        f.write(physical_location_mechanisms_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/physical_location_mechanism.md", "r").read())

    # Confinement Mechanisms
    if questions_and_answers["Q5"].find("1") != -1 and (questions_and_answers["Q8"].find("2") != -1 or questions_and_answers["Q8"].find("3") != -1):
        if questions_and_answers["Q14"].find("1") != -1 or questions_and_answers["Q16"].find("1") != -1 or questions_and_answers["Q17"].find("1") != -1:
            report.write("\n")
            report.write("\n")

            confinement_mechanisms_definition = openai.Completion.create(
                model="text-davinci-003",
                prompt="Definition of Security Confinement Mechanisms in Cloud-based mobile apps.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            confinement_mechanisms_examples = openai.Completion.create(
                model="text-davinci-003",
                prompt="Enumerate examples of Secure Confinement Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
                max_tokens=500,
                n=1,
                best_of=2,
            )

            f = open("security_mechanisms/confinement_mechanisms.md", "w")
            f.write("# Confinement Mechanisms \n\n")
            f.write(confinement_mechanisms_definition.choices[0]["text"].strip())
            f.close()

            f = open("security_mechanisms/confinement_mechanisms.md", "a")
            f.write("\n\n")
            f.write("## Confinement Mechanisms Examples: \n\n")
            f.write(confinement_mechanisms_examples.choices[0]["text"].strip())
            f.close()

            report.write(open("security_mechanisms/confinement_mechanisms.md", "r").read())

    # Filtering mechanisms
    if questions_and_answers["Q14"].find("1") != -1 and questions_and_answers["Q16"].find("1") != -1 and questions_and_answers["Q17"].find("1") != -1:
        report.write("\n")
        report.write("\n")

        filtering_mechanism_definition = openai.Completion.create(
            model="text-davinci-003",
            prompt="Definition of Security Filtering Mechanism Mechanisms in Cloud-based mobile apps.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        filtering_mechanism_examples = openai.Completion.create(
            model="text-davinci-003",
            prompt="Enumerate examples of Secure Filtering Mechanism Mechanisms in a table by Security Requirement, Mobile Plataform, Mechanism, Description, OSI Layer, to use for cloud-based mobile apps in coding phase and runtime? Generate response in Markdown format.\n\n",
            max_tokens=500,
            n=1,
            best_of=2,
        )

        f = open("security_mechanisms/filtering_mechanism.md", "w")
        f.write("# Filtering Mechanism Mechanisms \n\n")
        f.write(filtering_mechanism_definition.choices[0]["text"].strip())
        f.close()

        f = open("security_mechanisms/filtering_mechanism.md", "a")
        f.write("\n\n")
        f.write("## Filtering Mechanism Mechanisms Examples: \n\n")
        f.write(filtering_mechanism_examples.choices[0]["text"].strip())
        f.close()

        report.write(open("security_mechanisms/filtering_mechanism.md", "r").read())

    report.close()
    mechanisms_convert_report()
    print("\n\n # Processing done! Check your security mechanisms in the SECURITY_MECHANISMS.pdf file")

"""
[Summary]: Method responsible for creating, printing and outputting the complete processing report
[Arguments]: No arguments
[Return]: No return
"""
def fullReport():

    get_requirements()
    get_security_best_practices()
    get_mechanisms()
    get_attack_models()
    get_security_test_recommendation()

    pdfs = ['SECURITY_REQUIREMENTS.pdf', 'GOOD_PRACTICES.pdf', 'SECURITY_MECHANISMS.pdf', 'ATTACKS_MAPPING.pdf', 'TEST_SPECIFICATION.pdf']

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write("FULL_REPORT.pdf")
    merger.close()
    print("\n\n *** Processing  done! See the full report requested in the FULL_REPORT.pdf file! ***\n\n")


if __name__ == "__main__":
    print("---")
    print("")
    print("#  Welcome to ")
    print("")
    print("#  SecD4CLOUDMOBILE ")
    print("")
    print("  The **SecD4CLOUDMOBILE** is a custom made program")
    print("  This program implements a questionnaire about the development")
    print("  of mobile cloud-based application and generate a report with secure development guides.")
    print("  It is part of the outputs of the doctoral thesis project entitled Systematization of the ")
    print("  Security Engineering Process in the Cloud and Mobile Ecosystem ")
    print("")
    print("## License")
    print("")
    print("  Developed by Francisco T. Chimuco and Pedro R. M. Inácio")
    print("  Department of Computer Science")
    print("  Universidade da Beira Interior")
    print("")
    print("  Copyright 2021 Francisco T. Chimuco and Pedro R. M. Inácio")
    print("")
    print("  SPDX-License-Identifier: Apache-2.0")
    print("")
    information_capture()

    print("")
    print("#############################################################")

    print("")
    print("")
    print("")

    exit(0)

# license Apache-2.0
