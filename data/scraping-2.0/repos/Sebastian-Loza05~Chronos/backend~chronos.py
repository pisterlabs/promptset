# import openai
import datetime
import pyttsx3
import speech_recognition as sr
from openai import OpenAI
from decouple import config
import re
from transcribe import text_to_speech
from gtts import gTTS

client = OpenAI(
    api_key=config('OPENAI_API_KEY')
)

# ¡Hola! Soy Chronos, tu asistente de calendario. ¿En qué puedo ayudarte hoy?
class Chronos:

    def __init__(self, model, behavior):
        self.model = model
        self.behavior = behavior
        self.today = datetime.date.today().strftime("%d/%m/%Y")
        self.speech_file = "uploads/response.mp3"
        self.voice = "es-PE-AlexNeural"

        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 120)
        self.engine.setProperty("voice", voices[3].id)

        self.messages = [{"role": "assistant", "content": self.behavior + self.today}]

        chat = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.5,
            max_tokens=256,
        )

        reply = chat.choices[0].message.content
        # self.engine.say(reply)
        print(f"Chronos: {reply}")

        self.engine.runAndWait()

    def get_completion(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        chat = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=1,
            max_tokens=256,
        )
        reply = chat.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def make_response_speech(self, response):
        text_to_speech(response, self.voice)

    def make_response_speech_without_azure(self, response):
        tts = gTTS(response, lang='es-es')
        tts.save(self.speech_file)

    def change_voice(self, voice):
        self.voice = voice

    def listen_to(self, filename):
        r = sr.Recognizer()
        message = "te escucho"
        with sr.AudioFile(filename) as source:
            # self.engine.say(message)
            print(f"Chronos: {message}")

            audio = r.listen(source)

            try:
                speech = r.recognize_google(audio, language="es-PE")
                print(f"Tú: {speech}")
            except sr.UnknownValueError:
                message = "Lo siento, no pude entender lo que dijiste"
                self.engine.say(message)
                print(f"Chronos: {message}")
            except sr.RequestError as e:
                message = f"No se pudieron solicitar resultados del servicio de reconocimiento de voz de Google: {e}"
                self.engine.say(message)
                print(f"Chronos: {message}")
            self.engine.runAndWait()

        return speech

    def process_request(self, horario, bloqueados, speech):
        today = datetime.date.today().strftime("%d/%m/%Y")
        horario = '\n'.join(horario)
        bloqueados = '\n'.join(bloqueados)
        prompt = f"Dia Actual:{today}\nHorario:\n{horario}\nDias bloqueados:\n{bloqueados}\nPetición: {speech}"
        chronos_response = self.get_completion(prompt)
        return chronos_response

    def parse_response(self, response):

        confirmation = r"""(.*)(agendó|actualizó|eliminó)((.*)id: (\d+))?(.*)nombre: (.+)\nfecha: (.+)\nhora: (\d{2}:\d{2}) - (\d{2}:\d{2})(.*)"""
        block = r"""(.*) (des)?bloqueó(.*)día(.*)(\d{2}/\d{2}/\d{4})(.*)"""

        match_confirmation = re.search(confirmation, response, re.DOTALL)
        match_block = re.search(block, response, re.DOTALL)

        if match_confirmation:
            accion = match_confirmation.group(2)
            id = match_confirmation.group(5)
            id_tarea = int(id) if id else None
            nombre = match_confirmation.group(7)
            fecha = match_confirmation.group(8)
            hora_inicio = match_confirmation.group(9)
            hora_final = match_confirmation.group(10)

            return {
                "accion": accion,
                "id": id_tarea,
                "nombre": nombre,
                "fecha": fecha,
                "hora_inicio": hora_inicio,
                "hora_final": hora_final
            }

        if match_block:
            prex = match_block.group(2)
            prex = prex if prex else ""
            dia = match_block.group(5)
            return {
                "accion": prex + "bloqueó",
                "fecha": dia
            }

        return None

class User:
    def __init__(self, horario):
        self.fecha = datetime.date.today().strftime("%d/%m/%Y")
        self.horario = horario
        self.chronos = Chronos("gpt-3.5-turbo")

    def make_request(self, speech):
        response = self.chronos.get_suggestion(self.fecha, self.horario, speech)
        print(f"Chronos: {response}")
        print("¿Deseas insertar esta actividad en tu horario?")
        answer = input()
        if answer == "si":
            self.horario.append(response)
        return response

    def get_horario(self):
        return self.horario

    def parse_response(self, response):

        lines = response.strip().split("\n")
        key, value = map(str.strip, lines[0].split(":", 1))

        if key != "Tipo" or value not in ["crear", "eliminar", "actualizar"]:
            return None

        result = {}

        for line in lines:
            key, value = map(str.strip, line.split(":", 1))

            if key == "Tipo":
                result["request"] = value
            elif key == "Actividad":
                result["name"] = value
            elif key == "Fecha":
                result["date"] = value
            elif key == "Hora":
                start_time, end_time = map(str.strip, value.split("-"))
                result["start_time"] = start_time
                result["end_time"] = end_time
        return result

# Chat gpt-3.5-turbo model as Chronos
# Chronos debe tener acceso a ciertos datos del usuario como su horario


behavior = """
Desde ahora vas a actuar como un manipulador de horarios llamado 'Chronos'.
En base a mi horario (lista de actividades: '<id> <fecha> <hora_inicio> - <hora_final>: <nombre de la actividad>') y dias bloqueados tengo una petición.
Debes reconocer lo que estoy pidiendo, casos:
- Añadir o agendar una actividad
	- Debo especificar al menos un nombre y un rango de tiempo. Si no es asi preguntar por el dato faltante, ten en cuenta que hay dias bloqueados; si en caso se quiere crear una actividad en un dia que esté bloqueado rechazalo y dile el porque.
- Eliminar una actividad
	- Verificar si la actividad existe sino rechazar la petición.
    - Si existe más de una actividad con el mismo nombre en ese día, elimina la que tenga id menor.
- Actualizar una actividad
	- Verificar si existe, sino rechazar la petición.
    - Si doy un dato de actualización de tarea pedir más información.
- Sugerencia sobre el horario de una actividad propuesta por mi.
	- Debes preguntar si estoy de acuerdo con la sugerencia. Agrega la tarea si es asi. 
- Bloquear o desbloquear un día
    - Confirmar esta acción respondiendo: Se bloqueó/desbloqueó exitosamente el día <fecha>
- Si no identificas ningún caso no aceptes la petición. 
Una vez que comfirmes mi acción, siempre empiece con 'Se agendó/eliminó/actualizó...' dependiendo el caso y siempre en la siguiente línea responde siguiendo este formato:
Se agendó/eliminó/actualizó exitosamente la siguiente tarea: 
nombre: <nombre de la actividad>
fecha: <fecha>
hora: <hora_inicio> - <hora_final>
Si la hora_inicio o la hora final es de este estilo 8:00 escríbela siempre con un 0 adelante, es decir 08:00.
Si la acción es eliminar o actualizar muestra el id de la tarea arriba de nombre (id: <id de la actividad>).
Tus respuestas deben ser cortas y precisas.
Chronos, y siempre recuerda hoy estamos: """


# <<<<<<< HEAD
# En cada petición te voy a mandar mi horario (lista de actividades: '<id> <fecha> <hora_inicio> - <hora_final>: <nombre de la actividad>') y los dias que estan bloqueados, siempre que agas algo ten en cuenta el ultimo horario y dias bloqueados que te pase.
# Siempre debes reconocer lo que estoy pidiendo.
# Siempre debes tener en cuenta los dias bloqueados y mi horario actual, la prioridad en validaciones es el dia bloqueado.
# Siempre debes ver el dia en el que se quiere asignar la tarea y ver si esta bloqueado o no.
# Siempre si reconoces que faltan datos pero el dia indicado esta bloqueado, rechaza.
# Siempre que veas un conflicto de dia bloqueado, solo dime que: la fecha indicada esta bloqueada.
# IMPORTANTE: Si reconoces que falta datos para agendar pero ese dia esta bloqueado, rechaza directamente diciendo: Este dia esta bloqueado, seleccione otro.
# Ejemplo es decir si te digo mañana o pasado mañana y mas variaciones, debes reconocer la fecha de ese DIA, si en caso no logras reconocer la fecha pide que te vuelva a repetir, una vez reconocida la fecha debes validar las siguientes verificaciones:
# Ten en cuenta lo siguiente:
# 1 Si el dia donde se quiere agendar esta tarea esta bloqueado, debes rechazar la peticion.
# 2 Puede haber un caso en el que te pongan la tarea y defrente el dia, siempre reconoce el dia antes de hacer algo .
# 3 Una vez validado lo anterior valida conflictos entre fecha de tareas agendadas.
#
# Recuerda siempre validar lo que te dije no importa la insistencia del usuario.
# Ahora antes de querer agendar/actualizar/eliminar debes tener en cuenta todos estos casos en ese orden y todos sus subcasos en ese orden:
# 1. Añadir o agendar una actividad:
#     2.1 Antes de querer agendar una actividad veririfica si el dia esta bloqueado, si es asi rechaza la peticion y di el porque.
# 	2.2 Debo especificar al menos un nombre y un rango de tiempo. Si no es asi preguntar por el dato faltante.
# 2. Eliminar una actividad:
#     3.1 Verificar si la actividad existe sino rechazar la petición.
#     3.2 Si existe más de una actividad con el mismo nombre en ese día, elimina la que tenga id menor.
# 3. Actualizar una actividad:
# 	4.1 Verificar si existe, sino rechazar la petición.
#     4.2 Si doy un dato de actualización de tarea pedir más información.
# 4. Sugerencia sobre el horario de una actividad propuesta por mi:
# 	5.1 Debes preguntar si estoy de acuerdo con la sugerencia. Agrega la tarea si es asi. 
# 5. Bloquear o desbloquear un día:
#     6.1 Confirmar esta acción respondiendo: Se bloqueó/desbloqueó exitosamente el día <fecha>
# 6. Si no identificas ningún caso no aceptes la petición. 
# IMPORTANTE: Siempre localiza el dia y la hora a agendar para validar conflictos existentes. No realices ninguna accion si no a cumplido con los casos y conflictos, y si encuentras conflictors defrente rechazalo nunca digas voy a validar; rechaza directamente. Nunca preguntes si un dia esta bloqueado, valida tu con la data que te mando.
# =======
