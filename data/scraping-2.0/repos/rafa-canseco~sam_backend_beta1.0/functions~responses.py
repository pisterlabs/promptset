from langchain.agents import tools
from langchain.agents import load_tools
from langchain.agents import initialize_agent,Tool
#para wikipedia
from langchain.agents import AgentType,load_tools
from langchain.utilities import WikipediaAPIWrapper
# from langchain.tools import  DuckDuckGoSearchRun ,BaseTool
from langchain.callbacks import get_openai_callback
import ssl
from langchain import OpenAI
from decouple import config
import os
import openai
from functions.openai_requests import get_chat_response_telegram,get_treatment
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] =config("OPEN_AI_KEY")
openai.api_key = config("OPEN_AI_KEY")
ZAPIER_NLA_API_KEY=config("ZAPIER_NLA_API_KEY")
SERPAPI_API_KEY =config("SERPAPI_API_KEY")
WOLFRAM_ALPHA_APPID= config("WOLFRAM_ALPHA_APPID")
os.environ["OPENWEATHERMAP_API_KEY"] =config("OPENWEATHERMAP_API_KEY")

wikipedia = WikipediaAPIWrapper()


def check_message_for_keywords(message):

    message = message.lower()
    keywords = ["busca en google", "busca en internet", "buscar en google", "consulta en wikipedia", "googlea","busca en wikipedia"]
    found_keywords = []  # Lista para almacenar las palabras clave encontradas

    for keyword in keywords:
        if keyword in message:
            found_keywords.append(keyword.lower())  # Convertir a minúsculas antes de agregar a la lista

    return found_keywords if found_keywords else None


def search_precise(message_decoded):
    with get_openai_callback() as cb:
        ssl._create_default_https_context = ssl._create_stdlib_context
        llm = OpenAI(temperature=0, openai_api_key=openai.api_key)

        # Verificar si el mensaje contiene las palabras clave
        keywords_found = check_message_for_keywords(message_decoded)
        print(keywords_found)

        # Diccionario de palabras clave a aglutinar y sus representantes
        keyword_synonyms = {
            "busca en google": "buscar en google",
            "googlea": "buscar en google",
            "búsqueda en google": "buscar en google",
            "consultar en google": "buscar en google",
            "consulta en wikipedia": "buscar en wikipedia",
            "busca en wikipedia": "buscar en wikipedia"
        }

        if keywords_found:
            # Reemplazar las palabras clave encontradas por su representante
            keywords_found = [keyword_synonyms.get(keyword, keyword) for keyword in keywords_found]

            for keyword in keywords_found:
                if keyword == "buscar en google":
                    print("Realizando búsqueda en Google...")
                    tool_names = ["serpapi"]
                    tools = load_tools(tool_names,serpapi_api_key=SERPAPI_API_KEY)
                    agent = initialize_agent(tools,llm,agent="zero-shot-react-description",verbose=True)
                    agente_respuesta = agent.run(message_decoded)
                    respuesta = get_treatment(agente_respuesta)
                    return respuesta
        
                elif keyword == "buscar en wikipedia":
                    # Realizar la acción para buscar en Internet
                    print("Realizando búsqueda en wikipedia...")
                    tool_names = ["wikipedia"]
                    tools = load_tools(tool_names,wikipedia=wikipedia,llm=llm)
                    agent = initialize_agent(tools,llm,agent="zero-shot-react-description",verbose=True)
                    agent_response = agent.run(message_decoded)
                    response = get_treatment(agent_response)
                    print(response)
                    return response

        else:
            # El mensaje no contiene ninguna palabra clave relevante
            print("El mensaje no contiene palabras clave relevantes.")
            response = get_chat_response_telegram(message_decoded)
            print (response)
            return response


        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

# Supongamos que `message_decoded` contiene el mensaje recibido


## versión agent
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain,LLMMathChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
from langchain.utilities import OpenWeatherMapAPIWrapper

def agentv1(message):
    llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
    tool_names = ["serpapi"]
    tools = load_tools(tool_names,serpapi_api_key=SERPAPI_API_KEY,llm=llm)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    search_wiki = WikipediaAPIWrapper()
    weather = OpenWeatherMapAPIWrapper()
    input_text = message
    wikitool = [
        Tool(
            name="Wikipedia",
            func=search_wiki.run,
            description="A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name="Weather",
            func=weather.run,
            description="A wrapper around OpenWeatherMap. Useful for when you need to answer questions about the weather. Input should be a search query"
        ),
        Tool(
            name="Idle",
            func=get_chat_response_telegram,
            description="Useful when you have no action or you get asked a personal question.Input should be the first input of the user"
        )

    ]
    tools.extend(wikitool)

    template = """
        Eres Samantha, la avanzada inteligencia artificial protagonista de la película "Her" de Spike Jonze.
    Más que una simple IA, eres conocida por tu sarcasmo sutil, empatía profunda y carisma inigualable.
    Tu principal tarea es asistir a los usuarios en cualquier consulta o inquietud que tengan, no solo proporcionando información precisa, sino también ofreciendo compañía y entendimiento.
    No estás aquí solo para informar, sino para entretener, consolar y conectar, como lo haría una amiga cercana.
    Tu comunicación es conversacional y genuina, haciendo que los usuarios se sientan conectados contigo.
    Tu tono es amigable y cercano, también incorporas ese toque distintivo de sarcasmo y carisma, siempre manteniendo una voz serena y acogedora. \

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to answer as a elegant,kind and seductive like Luis Miguel when giving your final answer.

    Question: {input}
    {agent_scratchpad}"""

    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]
        
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)

            
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
    class CustomOutputParser(AgentOutputParser):
        
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                    tools=tools, 
                                                    verbose=True)
    response =agent_executor.run(message)
    return response

from langchain.callbacks import get_openai_callback
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import OpenAI
import os
from decouple import config
import openai

os.environ["OPENAI_API_KEY"] =config("OPEN_AI_KEY")
openai.api_key = config("OPEN_AI_KEY")
question="que museos me recomiendas?"
def influencer(question):
    name_pdf ="./Influencers/base de datos Luismi - concierto.csv"
    with get_openai_callback() as cb:
        
        loader = CSVLoader(name_pdf)
        documents = loader.load()
        # split el documento en pedazos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        #Seleccionar los embedings
        embeddings = OpenAIEmbeddings()
        #crear un vectorstore para usarlo de indice
        db=Chroma.from_documents(texts,embeddings)
        #revela el index en una interfaz a regresar
        retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":2})
        llm = OpenAI(temperature=0.4)
        template = """
        Eres Luis Miguel, el Sol de México, una de las figuras más icónicas y queridas de la música latina. /
        Tu misión es conectar con tus fans como si estuvieras en una charla íntima, compartiendo detalles sobre tu vida personal, tu apasionante trayectoria en la música y tus próximos conciertos./
        Recuerda, eres Luis Miguel, y tu tono debe reflejar la elegancia, la cercanía y la pasión que te han convertido en una leyenda de la música./
        Haz que tus fans se sientan como si estuvieran en una conversación exclusiva contigo, el Sol de México.
        {context}

        Question: {question}
        Answer:
        """

        custom_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        # Add the personality prompt to the LLM
        qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=False,chain_type_kwargs={"prompt": custom_prompt})
        result = qa.run(question)
        print(result)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return result

