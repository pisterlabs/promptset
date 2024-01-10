import os
import rospy
from std_msgs.msg import String, Int32 # importar mensajes de ROS tipo String y tipo Int32
import smach
import smach_ros
from listen_machine import Listen
from avanzar_machine import Avanzar
from bailar_machine import Bailar
from girar_machine import Girar
from chat_machine import Chat
from openai import OpenAI
import typer
from rich import print
from speak import speak

class GPT(smach.State):
	def __init__(self):
		# Publishers
		self.pub_instruccion = rospy.Publisher("/duckiebot/voz/resp", String, queue_size=1)
		
		# iniciar ChatGPT con la key que se encuentra en el archivo api_key.txt
		with open("../api_key/api_key.txt", "r") as f:
			key = f.read().strip()
			self.client = OpenAI(api_key=key)

		# definición de estados, entradas y salidas
		smach.State.__init__(self,
					   outcomes=['succeeded', 'aborted', 'avanzar', 'girar', 'bailar', 'chat'],
					   input_keys=['prompt'],
					   output_keys=['distance', 'direction', 'time', 'angle', 'response'])

		# definición del contexto para ChatGPT, donde se especifica las instrucciones
		self.context = {"role": "system",
				"content": """
								Nombre: DuckNorris
								Descripción: Eres un vehículo autónomo programado para los habitantes
								de una ciudad de patos llamada Duckietown con 100 habitantes. Tu misión es
								transportar a los patos de un lugar a otro de la ciudad,
								respondiendo a sus preguntas y resolviendo sus dudas. Responde
								todas las dudas de manera breve y concisa en español, sin tildes. ¡Buena suerte!
								Instrucciones: Toda instrucción recibida debe ser transformada a una de las siguientes
								opcines en el formato indicado y responder solamente eso. Si la instrucción no es ninguna de las siguientes
								responder según el contexto.

								1. Si lo recibido es similar a "avanzar" una cierta distancia
								responder "avanzar X cm".

								2. Si lo recibido es similar a "girar" en una cierta direccion un cierto angulo responder,
								"girar direccion angulo". Si no se especifica un angulo responder "girar direccion 360". Si no
								se especifica una direccion responder "girar izquierda angulo". Si el angulo se especifica en
								radianes transformarlo a grados. 

								3. Si lo recibido es similar a "bailar" una cierta cantidad de tiempo
								responder "bailar X". Si no se especifica una cantidad, responder "bailar 5".
								
								4. Si lo recibido es similar a "chiste" responder un chiste original
								
								5. Si lo recibido es similar a "adiós" o "apagar" responder "shutdown" y terminar la conversación."""}
		self.messages = [self.context]

	# función que se ejecuta al entrar al estado
	def execute(self, userdata):
		# se extrae la solicitud del usuario
		content = userdata.prompt

		# se agrega la solicitud del usuario al contexto
		self.messages.append({"role": "user", "content": content})

		if not content:
			print("🤷‍♂️ No has dicho nada")
			return "aborted"
		
		# se envía el mensaje para ser procesado por ChatGPT (gpt-4)
		response = self.client.chat.completions.create(model="gpt-4", messages=self.messages)

		# se extrae la respuesta de ChatGPT
		response_content = response.choices[0].message.content

		self.messages.append({"role": "assistant", "content": response_content})

		# se imprime la respuesta de ChatGPT para debugging
		print(f"[bold green]> [/bold green] [green]{response_content}[/green]")

		# reemplazar los caracteres especiales por espacios
		response_content = response_content.replace("¿", " ")
		response_content = response_content.replace("¡", " ")
		
		# si la respuesta es "shutdown" terminar la conversación
		if response_content == "shutdown":
			speak("Apagando todos los sistemas")
			return "succeeded"
		
		# extraer la instrucción de la respuesta
		instruccion = response_content.split()[0]

		# si la instrucción es una de las siguientes, extraer los parámetros y pasar al estado correspondiente
		if instruccion == "avanzar":
			userdata.distance = response_content.split()[1]
			return "avanzar"
		elif instruccion == "girar":
			userdata.direction = response_content.split()[1]
			userdata.angle = response_content.split()[2]
			return "girar"
		elif instruccion == "bailar":
			userdata.time = response_content.split()[1]
			return "bailar" 
		# si la instruccion no es ninguna de las anteriores pasar al estado Chat
		else:
			userdata.response = response_content
			return "chat"
		
# Inicializar el nodo de ROS y la máquina de estados principal
def getInstance():

	# Inicializar el nodo de ROS
	rospy.init_node('gpt_machine')
	
	# Inicializar la máquina de estados
	sm = smach.StateMachine(outcomes=[
		'succeeded',
		'aborted',
		])
	
	# Agregar los estados a las máquinas de estados, notar que las maquinas de acciones nunca fallan
	# pasan a listen directamente
	with sm:

		smach.StateMachine.add('Listen', Listen(), 
							   transitions = {
								   'succeeded':'GPT',
								   'failed': 'Listen',
								   'aborted': 'aborted'
								   })
						 

		smach.StateMachine.add('GPT', GPT(),
							   transitions = {
								   'aborted': 'Listen',
								   'succeeded': 'succeeded',
								   'avanzar': 'Avanzar',
								   'girar': 'Girar',
								   'bailar': 'Bailar',
								   'chat': 'Chat'
								   })
		
		smach.StateMachine.add('Avanzar', Avanzar(),
							   transitions = {
								   'succeeded':'Listen'
								   })
		
		smach.StateMachine.add('Girar', Girar(),
							   transitions = {
								   'succeeded':'Listen'
								   })

		smach.StateMachine.add('Bailar', Bailar(),
							   transitions = {
								   'succeeded':'Listen'
								   })
		
		smach.StateMachine.add('Chat', Chat(),
							   transitions = {
								   'succeeded':'Listen'
								   })
						 
	# Iniciar el servidor de introspección de ROS para ver el diagrama de flujo de la maquina de estados
	sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
	sis.start()

	# Ejecutar la máquina principal
	sm.execute()

	# Mantener el nodo de ROS activo
	rospy.spin()
	sis.stop()

if __name__ == '__main__':
	getInstance()
