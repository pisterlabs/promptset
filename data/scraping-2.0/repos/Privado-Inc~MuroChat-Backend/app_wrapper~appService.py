import logging
from django.conf import settings
from app_wrapper.utils import parseToOpenAIFormat
import openai
from django.http import StreamingHttpResponse
from app_wrapper.commonService import TOKEN_LIMIT, WITH_LAST_N_MESSAGES, numTokensForGPT, streamParserHandleInitator

from chats.dao.ChatHistoryDao import TYPE_OF_MESSAGE

log = logging.getLogger(__name__)

def getAI_Response(modelInfo, message, anonymizedMessage, piiToEntityTypeMap, chatHistory, userId, chatId, isPushedToChatHistory = False):
    openai.api_key = modelInfo['secretKey']
    
    history = list()
    if chatHistory:
        for messageObj in chatHistory['messages'][WITH_LAST_N_MESSAGES:]:
            history.append({
                "role": "assistant" if messageObj['type'] == TYPE_OF_MESSAGE['GPT'] else "user",
                "content": messageObj['anonymizedMessage']
            })

    if message:
        history.append({"role": "user", "content": f"{anonymizedMessage}"})

    tokens = numTokensForGPT(history, modelInfo['modelVersion'])
    while tokens >= TOKEN_LIMIT:
        history = history[2:]
        tokens = numTokensForGPT(history, modelInfo['modelVersion'])
        

    def event_stream(chatId, isPushedToChatHistory):
        log.info(chatId, history)
        completion = openai.ChatCompletion.create(
            model=modelInfo["modelVersion"], 
            messages = history,
            stream = True
        )
        streamParser = streamParserHandleInitator(chatId, isPushedToChatHistory)
        for line in completion:
            chunk = line['choices'][0].get('delta', {}).get('content', '')
            yield from streamParser(chunk, userId, message, anonymizedMessage, piiToEntityTypeMap)
    # TODO: Not working on local
    # stream_thread = threading.Thread(target=list, args=(event_stream(chatId, isPushedToChatHistory),)) # Need to convert later to asynchronouse approach
    # stream_thread.start()

    return StreamingHttpResponse(event_stream(chatId, isPushedToChatHistory), content_type='application/json')

