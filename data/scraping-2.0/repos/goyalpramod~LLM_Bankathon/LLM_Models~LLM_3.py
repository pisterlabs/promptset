from dotenv import load_dotenv, find_dotenv
import openai
import os
import re
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


chat = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo-16k")

jd="SEO manager. High social marketing skills required"

system_prompt = f"""
You are an AI model that provides a questionnaire for a job interview based on job description.
A detailed job description will be provided to you in triple back ticks. 
The job description is ```{jd}```
Generate at least 10 questions with their toughness and toughness score.
Also provide answers in brief for them.
IMPORTANT do not reply with "As an AI model..." under any circumstances 
"""

human_message_example = """
    Python Full stack Developer
"""

AI_message_example = """
1. What is Full Stack development? (Easy, 4/10)

Full Stack development involves developing both the front end and back end of the web application/website at the same time. This process includes three layers:

    Presentation layer (frontend part responsible for user experience) 

    Business logic layer (backend part refers to the server side of the application)

    Database layer.

If you want to enrich your career and become a professional Full Stack Developer, then enroll in "Full Stack Developer Training" - This course will help you to achieve excellence in this domain.
2. What do Full Stack Web Developers do? (Mediocre, 6/10)

A Full Stack Web Developer is a person who is familiar with developing both client and server software. In addition to mastering CSS and HTML, they are also know how to program browsers, databases, and servers.

To fully comprehend the role of Full Stack developer, you must understand the web development components - front end  and back end

The front end comprises a visible part of the application in which the user interacts, while the back end includes business logic.

Server Side vs Client Side
Q3. Name a few Full Stack developer tools. (Tough, 9/10)

Some of the popular tools used by full-stack developers to make development more accessible and efficient are:

    Backbone
    Visual Studio Code
    WebStorm
    Slack
    Electron
    TypeScript
    CodePen
    GitHub

4. What skills do you need to be a full-stack developer? (Easy, 4/10)

 A Full Stack developer should be familiar with:

    Basic languages - Must be proficient in basic languages like HTML, CSS, and SQL.
    Front-end frameworks - BootStrap, AngularJS, VueJS, ReactJS, JavaScript, TypeScript, Python, Ruby, PHP 
    Back-end frameworks - Express, Django, NodeJS, Ruby on Rails
    Databases - MySQL, SQLite, Postgres, MongoDB, Cassandra, Apache storm, Sphinx
    Additional skills recommended - Git, Machine Learning, SSH, Linux Command, Data Structures, Character encoding.

5. Explain Pair Programming? (Easy, 1/10)

As the name suggests, Pair Programming is where two programmers share a single workstation. Formally, one programmer at the keyboard called the "driver" writes the code. The other programmer is the "navigator" who views each line of the code written, spell check, and proofread it. Also, programmers will swap their roles every few minutes and vice-versa.
6. What is CORS? (Tough, 10/10)

Cross-origin resource sharing (CORS) is a process that utilizes additional HTTP headers to tell browsers to provide a web application running at one origin. CORS accesses various web resources on different domains. Web scripts can be integrated using CORS when it requests a resource that has an external origin (protocol. Domain, or port) from its own.
7. What is Inversion of Control (IoC)? (Tough, 8/10)

Inversion of Control (IoC) is a broad term used by software developers for defining a pattern that is used for decoupling components and layers in the system. It is mostly used in the context of object-oriented programming. Control of objects or portions of a program is transferred to a framework or container with the help of Inversion of Control. It can be achieved using various mechanisms such as service locator pattern, strategy design pattern, factory pattern, and dependency injection.
Full Stack Developer Interview Questions for Freshers
8. What is Dependency Injection? (Tough, 9/10)

Dependency Injection is a design pattern by which IoC is executed. Injecting objects or connecting objects with other objects is done by container instead of by the object themselves. It involves three types of classes.

    Client class: It depends on the service class.
    Service class: It provides service to the client class.
    Injector class: It injects service class objects into the client class.

9. What is Continuous Integration? (Easy, 3/10)

Continuous Integration (CI) is a practice where developers integrate code into a shared repository regularly to detect problems early. CI process involves automatic tools that state new code's correctness before integration. Automated builds and tests verify every check-in.
10. What is multithreading and how it is used? (Tough, 8/10)

The main purpose of multithreading is to provide multiple threads of execution concurrently for maximum utilization of the CPU. It allows multiple threads to exist within the context of a process such that they execute individually but share their process resources.
11. How is GraphQL different from REST? (Tough, 9/10)

This is typically a difficult question to answer, but a good developer will be able to go through this with ease. The core difference is GraphQL doesn't deal with dedicated resources. The description of a particular resource is not coupled to the way you retrieve it. Everything referred to as a graph is connected and can be queried to application needs.

Watch this video on “Top 10 Highest Paying IT Jobs in 2021” and know how to get into these job roles.

12. List the ways to improve your website load time and performance. (Tough, 9/10)

There are quite a lot of possible ways to optimize your website for the best performance:

    Minimize HTTP requests.
    Utilize CDNs and remove unused files/scripts.
    Optimize files and compress images.
    Browser caching. 
    Apply CSS3 and HTML5.
    Minify JavaScript & Style Sheets. 
    Optimize caches.

13. What is the Observer pattern? (Tough, 8/10)

The purpose of the Observer pattern is to define a one-to-many dependency between objects, as when an object changes the state, then all its dependents are notified and updated automatically. The object that watches on the state of another object is called the observer, and the object that is being watched is called the subject.
14. What’s the difference between a Full Stack Engineer and a Full Stack Developer? (Tough, 9/10)

A Full-Stack engineer is someone with a senior-level role with the experience of a Full-Stack developer, but with project management experience in system administration (configuring and managing computer networks and systems). 
15. What is polling? (Tough, 8/10)

Polling is a method by which a client asks the server for new data frequently. Polling can be done in two ways: Long polling and Short Polling.

    Long polling is a development pattern that surpasses data from server to client with no delay. 
    Short polling calls at fixed delays and is based AJAX-based.
16. How can you prevent a bot from scraping a publicly accessible API? (Tough, 9/10)

If the data within the API is publicly accessible, then it's not possible to prevent data scraping completely. However, there is an effective solution that will deter most people/bots: rate-limiting (throttling).

Throttling will prevent a specific device from making a defined number of requests within a defined time. Upon exceeding the specified number of requests, 429 Too Many Attempts  HTTP error should be thrown.

Other possible solutions to prevent a bot from scrapping are:

    Blocking requests based on the user agent string
    Generating temporary “session” access tokens for visitors at the front end

17. What is RESTful API? (Easy, 4/10)

REST stands for representational state transfer. A RESTful API (also known as REST API) is an architectural style for an application programming interface (API or web API) that uses HTTP requests to obtain and manage information. That data can be used to POST, GET, DELETE, and OUT data types, which refers to reading, deleting, creating, and operations concerning services.
18. What is a callback in JavaScript? (Tough, 8/10)

A callback in JavaScript is a function passed as an argument into another function, that is then requested inside the outer function to make some kind of action or routine. JavaScript callback functions can be used synchronously and asynchronously. APIs of the node are written in such a way that they all support callbacks.
19. What do you mean by data attributes? (Tough, 8/10)

Data Attributes are used to store custom data private to the application or page. They allow us to store extra data on the standard, semantic HTML elements. The stored data can be used in JavaScript’s page to create a more engaging user experience.

Data attribute consists of two parts:

    Must contain at least one character long after the prefix "data-" and should not contain uppercase letters.
    An attribute can be a string value.
20. What does ACID mean in Database systems? (Tough, 8/10)

Acronym ACID stands for Atomicity, Consistency, Isolation, and Durability. In database systems, ACID refers to a standard set of properties that ensure database transactions are processed reliably. 
21. How is rolling deployment different from blue-green deployment? (Tough, 8/10)

    In a rolling deployment, a new version of the application gradually replaces the previous one. Upgrading the system takes a period of time, and both old and new versions will coexist without affecting user experience or functionality in that phase.

    In a blue-green deployment, two identical production environments work in parallel. One is a blue environment that runs the production environment by receiving all user traffic. Another one is the green environment which you want to upgrade. Both use the same database backend and app configuration. If you swap the environment from blue to green, then traffic is directed towards a green environment.

22. What is an Application server? (Tough, 8/10)

An application server is a software framework that allows the creation of both web applications and server environments. It contains a comprehensive service layer model and supports various protocols and application programming interfaces (API).
23. What is referential transparency? (Tough, 8/10)

Referential transparency is a term used in functional programming to replace the expression without changing the final result of the program. This means that whether the input used is a reference or an actual value that the reference is pointing to the program's behavior is not changed.
"""

def func_(data):
    store = chat(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message_example),
            AIMessage(content=AI_message_example),
            HumanMessage(content=data)
        ]
    )
    return store

def separator(store):
    text=store.content
    # Regular expression to split the text into questions with their scores and answers
    question_pattern = r"(\d+)\. (.*?\?) \((.*?)\)\s+(.*?)\n\n"

    # Finding all matches using regular expression
    matches = re.findall(question_pattern, text, re.DOTALL)

    # Creating a list of dictionaries to store questions, answers, difficulty, and score
    questions_with_scores_and_answers = [
        {'Index': match[0], 'Question': match[1], 'Difficulty': match[2], 'Answer': match[3]} for match in matches]

    # Printing the questions with their respective difficulty levels, scores, and answers
#     for question_info in questions_with_scores_and_answers:
#         print(f"Question: {question_info['Question']}")
#         print(f"Difficulty: {question_info['Difficulty']}")
#         print(f"Index: {question_info['Index']}")
#         print(f"Answer: {question_info['Answer']}\n")
    return (questions_with_scores_and_answers)
# store = func_()
# print(store.content)
#
# print(separator(store))