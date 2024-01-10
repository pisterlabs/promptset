# Python libraries
import openai
import urllib.request
import os
import datetime
import requests
import base64
from tinytag import TinyTag
from math import ceil
from PIL import Image

# Custom modules
from main import *
from src.modules import security, db, stats, logtool
from src.env.app_secrets_env import openaiToken, fileKey
from src.env.app_public_env import maxTokensBotResponseGeneral, configBotResponses, voiceChoice

# Get OpenAI token
openai.api_key = openaiToken


'''
-------------------------------------------------------
GENERAL FUNCTIONS
-------------------------------------------------------
'''
def GetCurrentDatetime():
    datetimeCheck = datetime.datetime.now()
    formattedDatetime = datetimeCheck.strftime("%Y-%m-%d %H:%M:%S")
    return formattedDatetime


'''
-------------------------------------------------------
VOICE NOTE FUNCTIONS
-------------------------------------------------------
'''

def SpeechToText(userVoicePath):
    audioFile= open(userVoicePath, "rb")
    transcript = openai.audio.transcriptions.create(
    model="whisper-1", 
    file=audioFile
    )
    return transcript


def TextToSpeech(userVoiceNoteTranscripted, botVoicePath, voiceChoice):
    response = openai.audio.speech.create(
    model="tts-1",
    voice=voiceChoice,
    input=userVoiceNoteTranscripted,
    )

    response.stream_to_file(botVoicePath)


def AudioTranscriptProcessor(userVoiceNoteTranscripted):
    if userVoiceNoteTranscripted.lower().startswith("image") or userVoiceNoteTranscripted.lower().startswith("y mage"):
        result = "image"
    else:
        result = "text"
    
    return result


@security.UsersFirewall
async def VoiceInput(update: Update, context: CallbackContext) -> None:

    # Get current datetime for database tasks
    currentDateTime = GetCurrentDatetime()

    # Get basic info about the voice note file and prepare it for downloading
    userVoiceNoteId = await context.bot.get_file(update.message.voice.file_id)

    userVoicePath = f"src/temp/user_voice_note-{update.message.from_user.username}-{update.message.chat_id}.mp3"

    # Download the voice note as a file
    await userVoiceNoteId.download_to_drive(userVoicePath)

    # Get the duration of the voice note
    audio = TinyTag.get(userVoicePath)
    audioDuration = ceil(audio.duration / 60)

    # Calc user voice note tokens
    stats.StatsNumTokensWhisper(update.message.from_user.username, update.message.from_user.id, audioDuration)

    # Transcript voice note file to text
    audioTranscript = SpeechToText(userVoicePath)

    # Generate new Fernet Key
    fernetFileKey = security.GenerateFernetKey(fileKey)

    # Encrypt user voice note
    security.EncryptFile(userVoicePath, fernetFileKey)

    # Remove user voice note
    os.remove(userVoicePath)

    # Process the previous transcription to check if a specific voice command was pronounced
    transcriptProcessed = AudioTranscriptProcessor(audioTranscript.text)

    # Performs different actions if a specific voice command was detected
    if transcriptProcessed == "text":

        botAudioReply = GenerateTextReply(update.message.from_user.username, audioTranscript.text, update.message.from_user.id, update.message.chat_id, configBotResponses["Identity"], configBotResponses["Temperature"], viaInput="voice", option="tts")

        stats.StatsNumTokensTts(update.message.from_user.username, update.message.from_user.id, botAudioReply)

        botVoicePath = f"src/temp/bot_voice_note-{update.message.from_user.username}-{update.message.chat_id}.mp3"

        TextToSpeech(botAudioReply, botVoicePath, voiceChoice)

        await update.message.reply_voice(botVoicePath)

        # Encrypt bot voice note
        security.EncryptFile(botVoicePath, fernetFileKey)

        # Remove bot voice note
        os.remove(botVoicePath)
    
    elif transcriptProcessed == "image":
        await update.message.reply_photo(GenerateImageReply(update.message.from_user.username, audioTranscript.text[5::], update.message.from_user.id, update.message.chat_id, viaInput="voice", viaOutput="image"))


'''
-------------------------------------------------------
TEXT FUNCTIONS
-------------------------------------------------------
'''

def FormatCompletionMessages(userId, chatId, identity, option="prerequest"):

    results = db.GetUserMessagesToReply(userId, chatId)
    resultsFormatted = eval(str(results).replace("None", "'None'"))

    conversationFormatted = [{"role": "system", "content": identity}]
    for row in resultsFormatted:
        conversationFormatted.append({"role": "user", "content": row[2]})
        conversationFormatted.append({"role": "assistant", "content": row[10]})

    if option == "prerequest":
        conversationFormatted.pop()

    return conversationFormatted


def GenerateTextReply(username, prompt, userId, chatId, identity, temp, viaInput="text", viaOutput="text", option="gpt"):

    currentDateTime = GetCurrentDatetime()

    db.InsertUserMessage(username, prompt, userId, chatId, viaInput, viaOutput, currentDateTime)

    messagesFormatted = FormatCompletionMessages(userId, chatId, identity)
    logtool.userLogger.info(f'{username} sent a message')

    completions = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messagesFormatted,
        max_tokens=maxTokensBotResponseGeneral,
        n=1,
        stop=None,
        temperature=float(temp)
    )

    answerProvided = completions.choices[0].message.content

    db.InsertAssistantMessage(answerProvided, username, userId, chatId , viaInput, viaOutput, currentDateTime)

    messagesFormattedPost = FormatCompletionMessages(username, chatId, identity, option="postrequest")

    if option == "gpt":
        stats.StatsNumTokensGpt(username, userId, messagesFormattedPost)

    logtool.userLogger.info(f'Jepetobot replied a {option} message')

    return answerProvided


@security.UsersFirewall
async def TextInput(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    if update.message.text.startswith("IMAGE:"):
    #Reply a dalle image
        await update.message.reply_photo(GenerateImageReply(update.message.from_user.username, update.message.text.replace("IMAGE:",""), update.message.from_user.id, update.message.chat_id, viaInput="text", viaOutput="image"))

    else:
    # Reply the user message.
        await update.message.reply_text(GenerateTextReply(update.message.from_user.username, update.message.text, update.message.from_user.id, update.message.chat_id, configBotResponses["Identity"], configBotResponses["Temperature"], viaInput="text", viaOutput="text"))


@security.UsersFirewall
async def TextInputInline(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """Handle the inline query. This is run when you type: @botusername <query>"""

    query = update.inline_query.query

    if query == "":
        return

    results = [

        InlineQueryResultArticle(
            id="1",
            title="ReplyInline",
            description="Click here to get an answer",
            thumbnail_url="https://raw.githubusercontent.com/Alexvidalcor/jepetobot/master/src/images/Readme-logo2.jpg",
            input_message_content=InputTextMessageContent(query)
        )
    ]

    await update.inline_query.answer(results)


'''
-------------------------------------------------------
IMAGE FUNCTIONS
-------------------------------------------------------
'''

def GenerateImageReply(username, promptUser, userId, chatId, viaInput="text", viaOutput="image"):
    try:

        # Get current datetime for database tasks
        currentDateTime = GetCurrentDatetime()    

        db.InsertUserMessage(username, promptUser, userId, chatId, viaInput, viaOutput, currentDateTime)
        stats.StatsNumTokensDalle(username, userId)

        responseImage = openai.images.generate(
            model="dall-e-3",
            prompt=promptUser,
            n=1,
            size="1024x1024",
            quality="standard"
        )
 
        logtool.userLogger.info(f'Jepetobot replied a {viaOutput} via {viaInput}')

        return responseImage.data[0].url

    except openai.BadRequestError:
        return "src/images/App-image1.png"
    

def TransformToBase64(imagePath):

    with open(imagePath, "rb") as imagenFile:
        imagenData = imagenFile.read()

    # Convierte la imagen a base64
    imageBase64 = base64.b64encode(imagenData).decode('utf-8')

    return imageBase64


@security.UsersFirewall
async def ImageInput(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    try:
        # Get image id
        imageId = update.message.photo[-1].file_id
        imageReceived = await context.bot.get_file(imageId)

        # Save image in local
        userImagePath = f'src/temp/user_image-{update.message.from_user.username}-{update.message.chat_id}.jpg'
        await imageReceived.download_to_drive(userImagePath)

        # Convert image to base64
        imageBase64 = TransformToBase64(userImagePath)

        # Retrieve the text received alongside the image
        captionImageText = update.message.caption

        # Register vision tokens
        stats.StatsNumTokensVision(update.message.from_user.username, update.message.from_user.id)

        # Vision model in use via api
        responseVision = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "system", 
            "content": configBotResponses["Identity"]
            },
            {
            "role": "user",
            "content": [
                {"type": "text", "text": captionImageText},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{imageBase64}",
                    "detail": "high"
                },
                }
            ],
            }
        ],
        max_tokens=maxTokensBotResponseGeneral,
        )

        # Generate new Fernet Key
        fernetFileKey = security.GenerateFernetKey(fileKey)

        # Encrypt user image
        security.EncryptFile(userImagePath, fernetFileKey)

        # Remove bot voice note
        os.remove(userImagePath)

        # Bot response
        visionAnswerProvided = responseVision.choices[0].message.content
        await update.message.reply_text(visionAnswerProvided)
    
    except openai.BadRequestError:
        await update.message.reply_text("What do you want to know about the photo? Please enter your request in the 'description' field when sending me the image.")