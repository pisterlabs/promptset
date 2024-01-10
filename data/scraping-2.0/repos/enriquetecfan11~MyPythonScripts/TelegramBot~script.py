# -*- coding: utf-8 -*-
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Application, ContextTypes, MessageHandler
import logging

import os
from dotenv import load_dotenv
load_dotenv()

# IA Imports
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI

from translate import Translator
translator = Translator(to_lang="es")

pdf_path = "./documents/CursosTYC-GIS.pdf"

loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()

pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)

# Log con los errores que apareceran por pantalla.
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

def error_callback(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

# Start Command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Send a message when the command /start is issued."""
  await update.message.reply_text(f"Hola {update.effective_user.first_name}")
  await update.message.reply_text(f"Bienvido al bot de TYCGIS preguntame lo que quieras y te respondere sobre nuestra empresa.")
  await update.message.reply_text(f"Para mas informacion escribe /ayuda")

# Definir una funcion para manejar los mensajes de texto que recibe el bot.
# Esta funcion se ejecutara cada vez que el usuario envie un mensaje de texto.
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  # Obtener el mensaje del usuario
  mensaje_usuario = update.message.text

  if mensaje_usuario == "Hola" or mensaje_usuario == "hola" or mensaje_usuario == "Hola!" or mensaje_usuario == "hola!":
    await update.message.reply_text("Hola soy el bot de TYC GIS si quieres informacion sobre nuestros cursos solo preguntame.")
    return

  if mensaje_usuario == "Adios" or mensaje_usuario == "adios" or mensaje_usuario == "Adios!" or mensaje_usuario == "adios!":
    await update.message.reply_text("Adios, espero verte pronto. No olvides de visitar nuestra pagina web https://tycgis.com")
    return

  if mensaje_usuario == "quien eres" or mensaje_usuario == "Quien eres":
    await update.message.reply_text("Soy el bot de TYC GIS, si quieres informacion sobre nuestros cursos solo preguntame. Para mas informacion con el comnando /cursos puedes ver os cursos que tenemos disponibles.")
    return


  # Obtener la respuesta del bot
  respuesta = pdf_qa({"question": mensaje_usuario, "chat_history": ""})


  # Traducir la respuesta del bot
  translated = translator.translate(respuesta["answer"])

  # Si la repuesta traducida es "QUERY LENGTH LIMIT EXCEEDED. MAX ALLOWED QUERY : 500 CHARS" enviar la repuesta original

  if translated == "QUERY LENGTH LIMIT EXCEEDED. MAX ALLOWED QUERY : 500 CHARS":
    await update.message.reply_text(respuesta["answer"])
    return

  # Responder al usuario
  # await update.message.reply_text(respuesta["answer"])
  await update.message.reply_text(translated)

  # Guardar los mensajes del usuario y las repuestas en un archivo de texto con el nombre del usuario
  with open(f"messages.txt", "a") as file:
    file.write(f"Pregunta: {mensaje_usuario}\n")
    file.write(f"Respuesta: {respuesta['answer']}\n")
    file.write(f"Respuesta traducida: {translated}\n")
    file.write("--------------------------------\n")

# Comando cursos para mostrar los cursos disponibles
async def cursos(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  await update.message.reply_text("Cursos disponibles: \n - Curso de QGIS \n- Curso de ArcGIS Pro\n- Curso de ArcGIS Online\n- Curso de ArcGIS Enterprise\n- Curso de ArcGIS API for Python\n- Curso de Python para ArcGIS\n")
  await update.message.reply_text("Para mas informacion sobre los cursos puedes visitar nuestra pagina web https://cursosgis.com/ o sigue hablando conmigo.")

# Comando para contactar con TYC-GIS
async def contacto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  await update.message.reply_text("Para contactar con TYC GIS puedes hacerlo en el correo info@tycgis.com")
  await update.message.reply_text("Si lo deseas hacer por telefono puedes con la sede central en España (Madrid) (+34) 910 325 482 ")

# Comando ayuda para mostrar los comandos disponibles
async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  await update.message.reply_text("Comandos disponibles: \n - /start: Comando de bienvenida \n- /cursos: Nuestro cursos \n- /ayuda: Es este comando \n- /contacto: Para contactar con nostros a traves de correo electronico o a traves de nuestro telefono \n")
  await update.message.reply_text("Para mas informacion sobre los cursos puedes visitar nuestra pagina web https://cursosgis.com/ o sigue hablando conmigo.")

# Inicializar el bot
def main() -> None:
  """Start the bot."""
  # Create the Application and pass it your bot's token.
  app = Application.builder().token(os.getenv('TELEGRAM_TOKEN')).build()

  # Commands
  app.add_handler(CommandHandler("start", start))
  app.add_handler(CommandHandler("cursos", cursos))
  app.add_handler(CommandHandler("ayuda", ayuda))
  app.add_handler(CommandHandler("contacto", contacto))

  # Messages
  app.add_handler(MessageHandler(None, echo))

  app.run_polling()

if __name__ == '__main__':
  main()
