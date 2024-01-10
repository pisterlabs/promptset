#clase de busuqeda
import openai
import re
import time
from datetime import datetime
import json

#Utiliza la(s) siguiente(s) clase(s)
from script_taller import taller

class busqueda: 

  #Definimos valores de la búsqueda
  def __init__(self, texto):
    self.texto = texto

  
  #Función para crear una busqueda
  def crearTaller(self):

    #LLave para interactuar con la api de chat gpt
    openai.api_key = ""
  
    attributes = [["Tema"],["Duración"],["Cupos"],["Modalidad","NULL"],["Fecha","NULL"],["Hora","NULL"],["Nombre","NULL"],["Recinto","NULL"]]

    #Le pedimos una descripción del taller que se quiere buscar
    description = self.texto

    #Creamos la consulta a chat gpt para que nos retorne datos del taller
    inicio = time.time()
    question = '"' + description.descripcion + '"' + 'De esta descripción de un taller, retorna un string con la temática del taller. El tema debe ser una frase corta, o incluso una palabra. Si no puedes deducir el tema, o de que se trata, retorna NULL'
    prompt = openai.Completion.create(engine="text-davinci-003",
                              prompt=question,
                              max_tokens = 2048)

    regex_temaynombre = "([a-zA-zÑñÁáÉéÍíÓóÚúÜü]+\s*)+"
    Tema = re.search(regex_temaynombre,prompt.choices[0].text).group()
    attributes[0].append(Tema)


    attributes

    question = '"' + description.descripcion + '"' + 'De esta descripción de un taller, retorna un string con el siguiente formato "duracionTaller;cuposTaller". Si en alguna parte menciona la duración que tendría el taller, entonces retorna el número de horas a las cuales corresponde (solo retorna un número) en el campo duracionTaller. Si por el contrario no se menciona, retorna NULL en ese campo.'
    question += ' Si en alguna parte menciona cuantos cupos tendra el taller (osea cuantas personas podran participar), retorna el numero de cupos (solo retorna un numero) en el campo cuposTaller, si no se menciona entonces retorna NULL en ese campo. Recuerda que no debes escribir el nombre del campo, ni poner en tu respuesta algo como "Duracion del taller: 2 horas", si no mas bien solo debes retornar un numero o NULL en cada campo.'
    question += ' Ejemplos de outputs correctos son "3;20", "NULL;25", etc. No deduzcas duracion del taller ni cupos si no lo dice explicitamente o no es obvio, solo extrae la información si es que puedes encontrarla dentro de la descripción.'
    prompt = openai.Completion.create(engine="text-davinci-003",
                              prompt=question,
                              max_tokens = 2048)

    regex_duracionycupos = "([A-Z]+|(\d+(\.)\d+|\d+));([A-Z]+|\d+)"
    Duracion = re.search(regex_duracionycupos,prompt.choices[0].text).group().strip().split(';')[0]
    Cupos = re.search(regex_duracionycupos,prompt.choices[0].text).group().strip().split(';')[1]
    attributes[1].append(Duracion)
    attributes[2].append(Cupos)

    dicc = {
      "Tema": attributes[0][1],
      "Duracion" : attributes[1][1],
      "Cupos" : attributes[2][1],
      "Modalidad" : "NULL",
      "Fecha" : "NULL",
      "Hora" : "NULL",
      "Nombre" : "NULL",
      "Recinto" : "NULL"
    }

    print(dicc)

    return dicc

'''
    #Guardamos resultado en variable promt
    all_attributes = False
    nombre_ready = False

    while not all_attributes:
      for attribute in attributes:
        if attribute[1] == 'NULL':

          if attribute[0] == 'Duración':
            Duracion = input('\nMuy bien, ¿Cuánto tiempo tienes planeado que dure el taller?\nEscribe el número correspondiente a la cantidad de horas por favor.\nPor ejemplo, si tienes pensado que dure media hora, escribe 0.5, o si el taller va a durar 2 horas, escribe 2: ')
            attribute[1] = Duracion.strip()

          if attribute[0] == 'Cupos':
            Cupos = input('\n¿Y para cuantas personas tienes pensado que sea el taller?\nEscribe el número estimado: ')
            attribute[1] = int(Cupos)

          if attribute[0] == 'Modalidad':
            Modalidad = input('\n¿En qué modalidad te gustaría que se realizace el taller?\nEscribe Presencial u Online dependiendo de tu preferencia: ')
            attribute[1] = Modalidad.strip()

          if attribute[0] == 'Fecha':
            Fecha = input('\n¿Y en que fecha se realizaría este taller?\nEscribela en formato DD-MM-AAAA por favor: ')
            attribute[1] = Fecha.strip()

          if attribute[0] == 'Hora':
            Hora = input('\n¿A qué hora comenzaría el taller?\nEscribelo en formato 24 horas, es decir 18:30, 9:15, etc: ')
            attribute[1] = Hora.strip()

          if attribute[0] == 'Tema':
            Tema = input('\nOk, ¿De que te gustaría que se tratase el taller?: ')
            attribute[1] = Tema.strip()
          elif attribute[0] == 'Nombre':
            Buen_nombre = ''
            while not nombre_ready:
              if Buen_nombre == '':
                question = '"' + description + '"' + 'De esta descripción de un taller, cuya temática es '+attributes[0][1]+ ', inventa un nombre para el taller sin usar caracteres especiales y retornalo (no retornes Nombre del taller: blablabla, solo retorna el nombre a secas)'
                fin = time.time()
                print('\nEstoy pensando en un nombre para tu taller...')
                if fin-inicio < 60:
                  time.sleep(60 - (fin-inicio))
                prompt = openai.Completion.create(engine="text-davinci-003",
                                          prompt=question,
                                          max_tokens = 2048)

                Nombre = re.search(regex_temaynombre,prompt.choices[0].text).group().strip()
              texto = '\nSe me ocurre que tu taller podría llamarse \"'+Nombre+'\", ¿Te gusta ese nombre? ¿O no mucho?\nEscribe 0 si te gustó, 1 si quieres que invente otro nombre, o 2 si quieres inventarlo tu mism@: '
              Buen_nombre = input(texto)

              if Buen_nombre.strip() == '0':
                attributes[6][1] = Nombre.strip()
                nombre_ready = True
                break

              elif Buen_nombre.strip() == '1':

'''              


'''
              elif Buen_nombre.strip() == '2':
                Nombre = input('\n¿Cómo se llamará el taller? Escribe el nombre que desees: ')
                attributes[6][1] = Nombre.strip()
                nombre_ready = True
                break

              while Buen_nombre.strip() != '0' and Buen_nombre.strip() != '2' and Buen_nombre.strip() != '1':
                print('\nIngresa una de las opciones pedidas por favor\n')
                Buen_nombre = input(texto)

          if attribute[0] == "Recinto":
            if attributes[3][1] not in ['online','ONLINE','Online']:
              Respuesta = input("\n¿En donde se realizará el taller? Ingresa el número correspondiente:\n(1) Oficinas Apprende\n(2) USM Casa Central\n(3) USM Sede San Joaquín\n(4) USM Sede Viña del Mar\n(5) USM Sede Vitacura\n(6) USM Concepción \n")
              while Respuesta.strip() not in ['1','2','3','4','5','6']:
                print("\nIngresa una opción válida por favor")
                Respuesta = input("\n¿En donde se realizará el taller? Ingresa el número correspondiente:\n(1) Oficinas Apprende\n(2) USM Casa Central\n(3) USM Sede San Joaquín\n(4) USM Sede Viña del Mar\n(5) USM Sede Vitacura\n(6) USM Concepción\n")
              if Respuesta.strip() == '1':
                attribute[1] = 'Oficinas Apprende'
              if Respuesta.strip() == '2':
                attribute[1] = 'USM Casa Central'
              if Respuesta.strip() == '3':
                attribute[1] = 'USM Sede San Joaquín'
              if Respuesta.strip() == '4':
                attribute[1] = 'USM Sede Viña del Mar'
              if Respuesta.strip() == '5':
                attribute[1] = 'USM Sede Vitacura'
              if Respuesta.strip() == '6':
                attribute[1] = 'USM Concepción'

      all_attributes = True

      #[["Tema"],["Duración"],["Cupos"],["Modalidad","NULL"],["Fecha","NULL"],["Hora","NULL"],["Nombre","NULL"],["Recinto","NULL"]]

  #Tiene que crear un taller
  return taller(attributes[6][1], attributes[0][1], attributes[1][1], attributes[4][1], attributes[5][1], attributes[2][1], attributes[3][1], attributes[7][1])
'''