from fastapi import APIRouter, Body
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os # for environment variables
from dotenv import load_dotenv # pip install python-dotenv
load_dotenv() # load environment variables from .env file

addaRouter = APIRouter()

@addaRouter.post("/create-proyect/")
async def create_proyect( project_name: str = Body(), project_sector: str = Body(), project_idea: str = Body(), languague: str = Body()):

    #initialize the model
    llm = ChatOpenAI (
        api_key= os.getenv("OPEN_AI_API_KEY"),
        model='gpt-4',
        temperature= 0.9,
        max_tokens= 300,
    )

    #create the proyect
    proyect_template  = ChatPromptTemplate.from_template(
      """
        Eres un sistema de ia que ayuda a emprendedores y empresarios
          a generar proyectos negocios ideas, producto y/o servicios, 
        mi sector de interes es {project_sector} y busco una idea que sea 
        un producto o un servicio o una combinacion de ambos , incluye una 
        breve evaluacion de viabilidad de la idea asi como su publico objetivo ,
        incluye ejemplos y casos de estudo relevantes, la idea base de la que 
        tienes que partir es {project_idea}, transforma esta idea par crear una mas llamativa, atractiva e innovadora, el nombre base del proyecto del que tienes que partir es {project_idea}
        Busco una respuesta concreta y detallada que no pase de  dos parrafos, en la respuesta incluye un nuevo nombre que refleje 
        la audiencia objetivo y el sector de la idea, el nombre debe ser atractivo y no debes justificar el porque del nombre, 
        responde en el idoma {languague}
       
        
    """
    )

    #action plan template
    action_plan_template = ChatPromptTemplate.from_template(
        """ 
        Eres un sistema de ia que ayuda a emprendedores y empresarios
        a generar proyectos negocios ideas, producto y/o servicios,
        mi sector de interes es {project_sector} y la idea que tengo es {idea}

        tu tarea es crar un roadmap detallado para la creacion y el lanzamiento del producto o servicio
        al mercado, incluye un plan de accion detallado con los pasos a seguir para la creacion del producto o servicio
        los pasos deben ser claros con una linea de tiempo definida y entregables en cada paso,
        debes incluir los requerimientos tecnicos y de infraestructura necesarios para la creacion del producto o servicio
        como servidores, bases de datos, paginas web, aplicaciones, entre otros
        responde en el idoma {languague} 

        """
    )

    first_prompt = LLMChain(llm=llm, prompt=proyect_template, output_key="idea")
    second_prompt = LLMChain(llm=llm, prompt=action_plan_template, output_key="action_plan")

    cadena = SequentialChain(
        chains=[first_prompt, second_prompt],
        input_variables=["project_name", "project_sector", "project_idea", "languague"],
        output_variables=["idea", "action_plan"],
        verbose=True

    )

    output = cadena({
        "project_name": project_name,
        "project_sector": project_sector,
        "project_idea": project_idea,
        "languague": languague
    
    })

    return {
        "idea": output["idea"],
        "action_plan": output["action_plan"]
    }
