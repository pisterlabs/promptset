import os
from flask import Flask, Blueprint, make_response
from flask_restx import Api, Resource, reqparse


from models import CheckList, Task, check_list_schema, task_schema

from config import app, db

from api_control import taskCtrlr, taskDto, createTaskCommand, updateTaskCommand
from api_control import listCtrlr, listDto, createListCommand, updateListCommand
from api_control import langchainCtrlr, createlangChainCommand

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain

# a list that will act as a database for this example
todosDB = []

# create a flask blueprint for API application
blueprint = Blueprint('api', __name__, url_prefix='/api')

# create an API application attached to the blueprint
# you can also attacht he API application directly to the flask application
# use doc=False to disable swagger UI page
# use validate=True to enable request validation by default in all api endpoints/resources
api = Api(blueprint,
          title="Example API application",
          description="An example API application using flask-restx",
          version="1.0",
          doc="/swagger/",
          validate=True
          )

# create a namespace endpoint / controller endpoint by specifying the route
# A controller route is repesented by a python class inheriting the "Resource" base class
# The HTTP request handling functions are defined using the get, post, put, delete methods in the class
# Create a request parser to handle query parameters
parser = reqparse.RequestParser()
parser.add_argument('listId', type=int, help='Filter tasks by list ID')

@taskCtrlr.route("/")
class TodosDisplay(Resource):
    @taskCtrlr.marshal_list_with(taskDto)
    def get(self):
        args = parser.parse_args()
        list_id_filter = args.get('listId')
        # this method handles GET request of the API endpoint
        # return all database todo objects in response        
        if list_id_filter:
            tasks = Task.query.filter_by(list_id=list_id_filter).all()
        else:
            tasks = Task.query.all()
       
        return tasks
        

    @taskCtrlr.expect(createTaskCommand)
    def post(self):
        # this method handles POST request of the API endpoint
        # create a todo object in the database using the JSON from API request payload
        payload = taskCtrlr.payload        
        list_id = payload.get("list_id")
        if list_id and CheckList.query.get(list_id):
            newTask = Task(value=payload["value"], order=payload["order"],
                           list_id=payload["list_id"])
            db.session.add(newTask)
            db.session.commit()
            return task_schema.dump(newTask)
        else:
            taskCtrlr.abort(404, f"List with id {list_id} does not exist")


# extract id variable of endpoint from URL segment for use in the request handling functions
@taskCtrlr.route("/<int:id>")
class Todo(Resource):
    @taskCtrlr.marshal_with(taskDto)
    def get(self, id):
        # this method handles GET request of the API endpoint
        # get the todo object based on id from request URL
        
        task = Task.query.get(id)
        if task is not None:
            print(task)            
            print(task.list_id)            
            return task_schema.dump(task)
        else:            
            taskCtrlr.abort(404, f"Task with id {id} does not exist")
            
    @taskCtrlr.expect(updateTaskCommand)
    @taskCtrlr.marshal_with(taskDto)
    def put(self, id):        
        task = Task.query.get(id)        
        if task:
            payload = taskCtrlr.payload
            list_id = payload["list_id"]
            list = CheckList.query.get(list_id)
            if list is None:
                taskCtrlr.abort(404, f"List with id {list_id} does not exist")             
            task.value = payload["value"]
            task.order = payload["order"]
            task.list_id = payload["list_id"]
            task.completed = payload["completed"]
            db.session.merge(task)
            db.session.commit()
            return task_schema.dump(task), 201
        else:
            taskCtrlr.abort(404, f"Task with id {id} does not exist")
            
    def delete(self, id):
        task = Task.query.get(id)

        if task:
            db.session.delete(task)
            db.session.commit()
            return make_response(f"{task.value} successfully deleted", 200)
        else:
            listCtrlr.abort(404, f"Task with id {id} does not exist")
    
api.add_namespace(taskCtrlr)

@listCtrlr.route("/")
class CheckListDisplay(Resource):
    @listCtrlr.marshal_list_with(listDto)
    def get(self):        
        list = CheckList.query.all()
        return list
        
    @listCtrlr.expect(createListCommand)
    def post(self):        
        payload = listCtrlr.payload        
        newList = CheckList(title=payload["title"])
        db.session.add(newList)
        db.session.commit()
        return check_list_schema.dump(newList)
        
@listCtrlr.route("/<int:id>")
class CheckListInd(Resource):
    @listCtrlr.marshal_with(listDto)
    def get(self, id):                
        list = CheckList.query.get(id)
        if list is not None:            
            return check_list_schema.dump(list)
        else:            
            listCtrlr.abort(404, f"List with id {id} does not exist")
    
    @listCtrlr.expect(updateListCommand)
    @listCtrlr.marshal_with(listDto)
    def put(self, id):        
        list = CheckList.query.get(id)

        if list:
            payload = listCtrlr.payload
            list.title = payload["title"]
            db.session.merge(list)
            db.session.commit()
            return check_list_schema.dump(list), 201
        else:
            listCtrlr.abort(404, f"List with id {id} does not exist")
            
    def delete(self, id):
        list = CheckList.query.get(id)

        if list:
            db.session.delete(list)
            db.session.commit()
            return make_response(f"{list.title} successfully deleted", 200)
        else:
            listCtrlr.abort(404, f"List with id {id} does not exist")
    
        
api.add_namespace(listCtrlr)

@langchainCtrlr.route("/")
class SendTaskToLangChain(Resource):
    @langchainCtrlr.expect(createlangChainCommand)
    def post(self):        
        payload = langchainCtrlr.payload                
        
        class CommaSeparatedListOutputParser(BaseOutputParser):
            """Parse the output of an LLM call to a comma-separated list."""


            def parse(self, text: str):
                """Parse the output of an LLM call."""
                return text.strip().split(", ")
        
        llm = OpenAI(openai_api_key=os.environ.get("SECRET_KEY_OPENAI"))        
        template = """Eres un asistente que genera una lista sin numeracion separada por comas.
        La cantidad maxima de elementos por lista es de 10.
        Los elementos de la lista deben estar ordenados.        
        Imagina que eres un expero en un productividad dentro de tres
        asteristicos vamos a escribirte el titulo de una lista de tareas y quiero que
        me digas que actividades debo realizar para completarla. 
        ***
        {data}
        ***
        """
        prompt = PromptTemplate.from_template(template)
        
        prompt.format(data=payload["prompt"])
        list = llm.predict(prompt.format(data=payload["prompt"])).strip().split(", ")
        #print("list", list)
        
        newList = CheckList(title=payload["prompt"])
        db.session.add(newList)
        db.session.commit()
        
        order = 1
        for task in list:
            newTask = Task(value=task, order=order,
                 completed=False, list_id=newList.id)
            order +=1
            db.session.add(newTask)
            db.session.commit()
        return make_response(f"List created successfully", 200)
        
        '''
        #Segunda opción, pero con menos calidad en la respuesta
        chain = LLMChain(
            llm=OpenAI(openai_api_key=os.environ.get("SECRET_KEY_OPENAI")),
            prompt=prompt,
            output_parser=CommaSeparatedListOutputParser()
        )
        print(chain.run(data=payload["prompt"]))'''
        
        
api.add_namespace(langchainCtrlr)