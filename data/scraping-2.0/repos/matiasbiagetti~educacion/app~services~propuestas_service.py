from fastapi import Depends

from app.services.cursos_service import CursosService
from app.services.openai_service import OpenAIService


class PropuestasService:
    def __init__(self, openai_service: OpenAIService = Depends(OpenAIService)):
        self.openai_service = openai_service

    def crear_propuesta(self, curso_codigo: int, tema: str, tipo: str):
        """
        Crea una propuesta de clase en base a un tema un tipo y una materia
        """

        texto = self.openai_service.crear_propuesta(curso_codigo, tema, tipo)
        print(texto)

        return texto

