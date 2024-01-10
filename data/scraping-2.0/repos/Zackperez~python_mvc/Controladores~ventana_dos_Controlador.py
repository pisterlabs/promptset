from unittest import result
from ventana_principal import *
from Modelos.ventana_dos_Modelo import *
from Vistas.ventana_dos_Vista import *
import tkinter as tk
import nlpcloud
import openai
import json

class Ventana_Dos_Controller:

    def __init__(self, root):
        self.model = Ventana_dos_Model()
        self.view = Ventana_dos_View(root)
        self.view.btnEnviar.config(command=self.analizar_sentimiento)
        self.view.btnspin.config(command=self.preguntas)

    def analizar_sentimiento(self):
        client = nlpcloud.Client("distilbert-base-uncased-finetuned-sst-2-english", "1c56cb1a8a4b5cb1079f2f2e0c89321585206468", gpu=False, lang="en")
        #Para analizar el sentimiento, se necesita que el ususario escriba en un campo y así usar client.sentiment()
        sentimiento_analizar = self.view.txtEntrada.get()
        res = client.sentiment(sentimiento_analizar)
        #Entrar dentro del JSON dentro de 'scored_labels' que retorna RES para sacar el sentimiento analizado
        a = res.get('scored_labels')
        listapy = a[0]
        listafinal = listapy.items()
        sentimiento_analizado = list(listafinal)[0][1]
        self.crear_traduccion_json("Analizador de sentimientos","Sentimiento a analizar:", sentimiento_analizar, "Sentimiento analizado:", sentimiento_analizado)
        self.view.lblresultado ['text'] = sentimiento_analizado #Muestra el sentimiento encontrado
        self.view.txtEntrada.delete(0,END)

    def preguntas(self):
        openai.api_key = ("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")
        #Se almacena el valor actual del SPINBOX
        cantidad_spinbox= self.view.spin_box.get()

        #Se guarda lo que trae el campo de la pregunta
        pregunta = self.view.txtEntrada_pregunta.get()

        #Se concatena la pregunta con la cantidad que se quiere mostrar
        pregunta_final= pregunta + cantidad_spinbox 

        #La respuesta está almacenada dentro de un JSON por el cual hay que entrar para obtener la respuesta
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=pregunta_final,
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        #Dentro del JSON que retorna, se encuentra dentro del primer puesto (choices) dentro de 'text'
        lista_generada = response.choices[0].text.strip() #Al principio hay un pequeño espacio por lo cual se usa .strip para eliminarlo

        #Se inserta el resultado dentro del widget txtRespuesta
        self.view.txtRespuesta.insert(END, lista_generada)
        self.view.txtEntrada_pregunta.delete(0,END)
        self.crear_traduccion_json("Generador de listas","Lista a generar:", pregunta, "Lista generada:" , lista_generada)
    
    def existe_historial(self):
        try:
            with open('historial.json') as archivo:
                return True
        except FileNotFoundError as e:
            return False

    def devolver_respuestas(self,nombre_servicio, espacio_1, valor_espacio_1, espacio_2, valor_espacio_2):
        diccionario_generado = {}
        diccionario_generado = {espacio_1 : valor_espacio_1, espacio_2 : valor_espacio_2}
        respuesta_final = {nombre_servicio: diccionario_generado}
        return respuesta_final

    def crear_traduccion_json(self, nombre_servicio, espacio_1, valor_espacio_1, espacio_2, valor_espacio_2):

        if self.existe_historial() == True:
            nuevos_datos = self.devolver_respuestas(nombre_servicio, espacio_1, valor_espacio_1, espacio_2, valor_espacio_2)
            with open("historial.json",encoding="utf-8") as archivo_json:
                datos = json.load(archivo_json)
                datos.append(nuevos_datos)
            with open("historial.json", 'w',encoding="utf-8") as archivo_json:
                json.dump(datos, archivo_json, indent=3, ensure_ascii=False)
                print("Se han añadido los siguientes datos al archivo " + archivo_json.name+"\n")
        else:
            with open("historial.json", 'w',encoding="utf-8") as archivo_json:
                historial = []
                historial.append(self.devolver_respuestas(nombre_servicio, espacio_1, valor_espacio_1, espacio_2, valor_espacio_2))
                json.dump(historial, archivo_json, indent=3, ensure_ascii=False)
                print(archivo_json.name+" creado exitosamente")
                print("Se han añadido los siguientes datos al archivo " + archivo_json.name+"\n")