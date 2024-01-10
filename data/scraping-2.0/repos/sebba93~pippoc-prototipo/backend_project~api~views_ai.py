import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
#from openai import OpenAI
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from .gpt_ia.gpt_ia import consulta_gpt
from rest_framework.response import Response
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = "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"

#client = OpenAI(api_key="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")

u1 = "En esta unidad examinaremos algunos conceptos básicos como trabajo, salud, riesgos profesionales, factores de riesgo o accidente de trabajo y enfermedad profesional, que nos permitirán descubrir cuál es el proceso por el que se llega a poner en peligro la salud de los trabajadores."

u11 = "La Organización Mundial de la Salud (OMS), define la salud como “el estado de completo bienestar físico, mental y social y no solamente la ausencia de enfermedad”. De la definición de la OMS, es importante resaltar el aspecto positivo, ya que se habla de un estado de bienestar y no solo de ausencia de enfermedad."

u12 = "Se entiende por trabajo cualquier actividad física o intelectual. El trabajo remunerado es un medio para satisfacer las necesidades humanas: la subsistencia, la mejora de la calidad de vida, la posición del individuo dentro de la sociedad, la satisfacción personal, etc."

u13 = "Es evidente que el trabajo y la salud están estrechamente relacionados, ya que el trabajo es una actividad que el individuo desarrolla para satisfacer sus necesidades, al objeto de disfrutar de una vida digna. También gracias al trabajo podemos desarrollarnos tanto física como intelectualmente. Salud laboral consiste pues, en promover y proteger la salud de las personas en el trabajo evitando todo aquello que pueda dañarla y favoreciendo todo aquello que genere bienestar, tanto en el aspecto físico como en el mental y social."

u14 = "Junto a esta influencia positiva del trabajo sobre la salud existe otra negativa, la posibilidad de perder la salud debido a las malas condiciones en las que se realiza el trabajo, y que pueden ocasionar daños a nuestro bienestar físico, mental y social (accidentes laborales, enfermedades...). Los elementos que influyen negativamente y relacionados con la seguridad y la salud de los trabajadores son los riesgos laborales. Por tanto, podríamos decir que un trabajador o trabajadora está expuesto a riesgo laboral en aquellas situaciones que pueden romper su equilibrio físico, psíquico o social. Para calificar un riesgo desde el punto de vista de su gravedad, se valorarán conjuntamente la probabilidad de que se produzca el daño y la severidad del mismo. Se entenderá como ≪riesgo laboral grave e inminente≫ aquel que resulte probable racionalmente que se materialice en un futuro inmediato y del que puedan derivarse daños graves para la salud. Existe otro concepto habitualmente relacionado con la prevención de riesgos y que frecuentemente se confunde al asemejarse al concepto de riesgo. Es el concepto de peligro: fuente de posible lesión o daño para la salud. Ejemplos de condiciones peligrosas: instalaciones inadecuadas o en mal estado, equipos, útiles, elementos o materiales defectuosos, resguardos y protecciones inadecuadas o inexistentes en máquinas o instalaciones, condiciones ambientales peligrosas (ej: por la presencia no controlada de polvo, gases, vapores, humos, ruidos, radiaciones, etc.), ausencia de delimitación de áreas de trabajo, de tránsito de vehículos, de personas, etc."

@csrf_exempt
def consult(request):
    unidad1 = "Unidad 1: " + u1 + "\n"
    unidad11 = "Unidad 1.1: " + u11 + "\n"
    unidad12 = "Unidad 1.2: " + u12 + "\n"
    unidad13 = "Unidad 1.3: " + u13 + "\n"
    unidad14 = "Unidad 1.4: " + u14 + "\n"
    msg = "Sean las unidades: \n" + unidad1 + unidad11 + unidad12 + unidad13 + unidad14 + "Mi pregunta es: " + str(request.body)
    respuesta = consulta_gpt(GPT_MODEL, msg)
    print(respuesta)
    return JsonResponse(respuesta)