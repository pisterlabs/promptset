from typing import Any, Optional, Type

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.schema import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from .base import ejecutar_sql
import os

def leer_informacion(file_path:str) -> str:
    # Inicializa la variable de cadena
    file_content = ""

    # Comprueba si el archivo existe
    if os.path.isfile(file_path):
        # Si el archivo existe, lo abre y lee el contenido
        with open(file_path, "r") as file:
            file_content = file.read()
        return file_content
    return None

# You can provide a custom args schema to add descriptions or custom validation
class QuerySchema(BaseModel):
    query: str = Field(description="Solo puede ser una consulta SQL entre (SELECT, UPDATE o INSERT) separadas por ';'.")

# You can provide a custom args schema to add descriptions or custom validation
class InfoSchema(BaseModel):
    query: str = Field(description="""Un nombre de una tabla base de datos. A elegir entre:
                      (camareros, teclas, mesas, infmesa, lineasmesa, arqueos, cierrecaja, tikcet,
                      ticketlinea, secciones, zonas, efectivo o gastos)""" )

   
class ExecSQLTools(BaseTool):
    name = "exec_sql"
    description = "Utiliza esta herramienta para ejecutar consultas del tipo [SELECT, INSERT, UPDATE]. Parametro: consulta SQL without coutes"
    
    args_schema: Type[QuerySchema] = QuerySchema
   
    
    def _run(self, query: str,  run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return  ejecutar_sql(query)
       

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self._run(query, run_manager)
 
class SearchInfoDBTools(BaseTool):
    name = "search_info_db"
    description = """Util para buscar informacion de la estructura de datos y ejemplos de consultas SQL de la tabla."""
    
    args_schema: Type[InfoSchema] = InfoSchema
    path_db_info: str


    def _run(self, query: str,  run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Especifica la ruta y el nombre del archivo
        file_name = query.strip()+".txt"
        file_path = os.path.join(self.path_db_info, file_name)
        info_str = leer_informacion(file_path)
        print(file_path, info_str)
        if info_str:
            return info_str
        else:
            return f"""La tabla {query} no exite. Elige entre:
                       (camareros, teclas, mesas, infmesa, lineasmesa, arqueos, cierrecaja, tikcet,
                       ticketlinea, secciones, zonas, efectivo o gastos).
                     """
        
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self._run(query, run_manager)
 
class EjemplosInfoDBTools(BaseTool):
    name = "ejemplos_info_db"
    description = """Util para buscar ejemplos de consulta sql."""
    
    args_schema: Type[InfoSchema] = InfoSchema
    path_db_info: str


    def _run(self, query: str,  run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Especifica la ruta y el nombre del archivo
        file_name = query.strip()+".ejemplos"
        file_path = os.path.join(self.path_db_info, file_name)

        info_str = leer_informacion(file_path)

        if info_str:
            return info_str
        else:
            return f"""La tabla {query} no exite. Elige entre:
                       (camareros, teclas, mesas, infmesa, lineasmesa, arqueos, cierrecaja, tikcet,
                       ticketlinea, secciones, zonas, efectivo o gastos).
                     """
        
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self._run(query, run_manager) 

class SharchEmbeddingsTools(BaseTool):
    name = "search_info_db"
    description = "Utiliza esta herramienta para ver ejemplos SQL y la estructura de datos."
    
    args_schema: Type[InfoSchema] = InfoSchema
    rt: Any

    def _run(self, query: str,  run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        relevant_text = ""
        for d in self.rt.get_relevant_documents(query):
            relevant_text += d.page_content
        return relevant_text
        
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self._run(query, run_manager)
    

