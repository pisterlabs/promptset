import pdb

from rest_framework import status
from app_wrapper.appService import getAI_Response
from app_wrapper.commonService import applyDataFirewall
from app_wrapper.llamaService import getAI_ResponseFromLlama
from chats.dao.ChatDao import TYPE_OF_AI, ChatDao
from chats.dao.LlmModelDao import LlmModelDao
from chats.dao.UserChatShareDao import UserChatShareDao
from chats.dao.ChatHistoryDao import TYPE_OF_MESSAGE, ChatHistoryDao
from utils.EmailClient import EmailClient
from utils.accessorUtils import getOrNone
from utils.cryptoClient import getCryptoClient
from utils.dateTimeUtils import convert_bson_datetime_to_string
from utils.paginationUtils import paginationMeta
from utils.responseFormatter import formatAndReturnResponse
import logging
import json
from bson.json_util import dumps
from django.core.paginator import Paginator
from bson.objectid import ObjectId
from users.models import User
from datetime import datetime, timedelta
from django.conf import settings
import openai

log = logging.getLogger(__name__)

chatDao = ChatDao()
chatHistoryDao = ChatHistoryDao()
userChatShareDao = UserChatShareDao()
llmModelDao = LlmModelDao()
crypto = getCryptoClient()

def createChat(userId, data, isUI):
    log.info('createChat')

    if "name" not in data:
        return formatAndReturnResponse({ "message": "Name is missing while creating chat"}, status=status.HTTP_200_OK, isUI=isUI)
    response = chatDao.createChat(userId, data["name"]) # Here need to get name from the ML which is context of first message
    if not response:
            if response is None or not response.acknowledged:
                formatAndReturnResponse({'message': 'Failed to create chat. May be the name is already exist.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    return formatAndReturnResponse({"message": 'chat created succesfully', "userId": userId }, status=status.HTTP_200_OK, isUI=isUI)
    
def getChats(userId, page, pageSize, searchTerm, isUI):
    log.info('getChats')
    
    data = json.loads(dumps(chatDao.getChats(userIds=[userId], searchTerm=searchTerm)))
    if data and len(data):
        paginator = Paginator(data, pageSize)
        page_obj = paginator.get_page(page)
        chats = list()
        for chat in page_obj:
            chats.append(chat)
        meta = paginationMeta(paginator, page_obj, pageSize)

        return formatAndReturnResponse(chats, status=status.HTTP_200_OK, isUI=isUI, pageInfo=meta)
    return formatAndReturnResponse({'message': 'No chats found for given user id ' + str(userId)}, status=status.HTTP_404_NOT_FOUND, isUI=isUI)

def getChatsByPeriod(userId, isUI):
    chats = {}
    periods = settings.CHAT_GROUPS
    chats['pinnedChats'] = json.loads(dumps(chatDao.getPinnedChats(userId)))
    chatsByDate = list()
    for period in periods:
        query = getQueryForPeriod(period)
        periodChats = json.loads(dumps(chatDao.getChats(userIds=[userId], period=query)))
        for chat in periodChats:
            chat["createdAt"] = chat["createdAt"]["$date"]
            chat["modifiedAt"] = chat["modifiedAt"]["$date"]
        chatsByDate.append({'title': period, 'chats': periodChats})

    chats['chatsByDate'] = chatsByDate
    return formatAndReturnResponse(chats, status=status.HTTP_200_OK, isUI=isUI)

def getQueryForPeriod(period):
    today = datetime.now()
    start_of_today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    seven_days_ago = start_of_today - timedelta(days=6)
    thirty_days_ago = start_of_today - timedelta(days=29)
    if period == 'Today':
        return {
            "modifiedAt": {
                "$gte": today.replace(hour=0, minute=0, second=0, microsecond=0),
                "$lt": today.replace(hour=23, minute=59, second=59, microsecond=999),
            }
        }
    elif period == 'Last 7 Days':
        return {
            "modifiedAt": {"$gte": seven_days_ago, "$lt": start_of_today}
        }
    elif period == 'Last 30 Days':
        return {
            "modifiedAt": {"$gte": thirty_days_ago, "$lt": seven_days_ago}
        }
    elif period == 'Others':
        return {
            "modifiedAt": {"$lt": thirty_days_ago}
        }

def createOrUpdateChatMessage(userId, chatId, data, isUI, isPushedToChatHistory = False):
    log.info('createOrUpdateChatMessage')
    if "message" not in data and not isPushedToChatHistory:
        return formatAndReturnResponse({'message': 'Message field is missing.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    chatHistory = json.loads(dumps(chatHistoryDao.getChatHistory(userId, chatId)))
    history = list()
    if chatHistory:
        for messageObj in chatHistory['messages']:
            history.append({
                    "role": "system" if messageObj['type'] == TYPE_OF_MESSAGE['GPT'] else "user",
                    "content": messageObj['message'],
                    "anonymizedMessage": messageObj['anonymizedMessage'],
                    "piiToEntityTypeMap": messageObj['piiToEntityTypeMap']
            })
    try:
        message = data.get("message", None)
        response = llmModelDao.getDefaultLlmModel()
        if response:
            response["secretKey"] = crypto.decrypt(response["secretKey"].encode('utf-8')).decode('utf-8')
            if not message:
                  lastUserMessage = chatHistory["messages"][-1]
                  anonymizedMessage = lastUserMessage["anonymizedMessage"]
                  piiToEntityTypeMap = lastUserMessage["piiToEntityTypeMap"]
            else:
                anonymizedMessage, piiToEntityTypeMap = applyDataFirewall(message, chatHistory.get("piiToEntityTypeMap", {}) if chatHistory else {})
            if response['type'] == TYPE_OF_AI['GPT']:
                aiResponse = getAI_Response(response, message, anonymizedMessage, piiToEntityTypeMap, chatHistory, userId, chatId, isPushedToChatHistory)
            else:
                aiResponse = getAI_ResponseFromLlama(response, message, anonymizedMessage, piiToEntityTypeMap, chatHistory, userId, chatId, isPushedToChatHistory)

        else:
            return formatAndReturnResponse({ 'message': 'No Model Selected'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, isUI=isUI)

    except Exception as e:
        log.error(e)
        print(e)
        return formatAndReturnResponse({ 'message': 'Failed to get response from Chat GPT'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, isUI=isUI)
    return aiResponse

def deleteChatMessage(userId, chatId, isUI):
    log.info('deleteChatMessage')
    if not chatId:
        return formatAndReturnResponse({'message': 'ChatId field is missing.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    chat = chatDao.deleteChat(userId, chatId)
    if chat:
        return formatAndReturnResponse({ 'message': 'Successfully deleted chatId ' + chatId}, status=status.HTTP_200_OK, isUI=isUI)
    return formatAndReturnResponse({'message': 'No chat found with chatId' + chatId}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)



def createMessages(userId, chatId, data, isUI):
    log.info('createChat ' + str(data))
    if "message" not in data:
        return formatAndReturnResponse({'message': 'Message field is missing.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    message = data["message"]

    isChatExist = chatDao.getChats(userIds=[userId]) and chatId
    response = None
    if not isChatExist:
        response = chatDao.createChat(userId, message[:24]) # Here need to get name from the ML which is context of first message
        if not response.acknowledged:
            return formatAndReturnResponse({'message': 'Failed to create chat. May be the name is already exist.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    finalChatId = chatId or str(response.inserted_id)

    (data, _) = chatHistoryDao.createOrUpdateChat(userId, finalChatId, message, 'USER_INPUT')
    (data, _) = chatHistoryDao.createOrUpdateChat(userId, finalChatId, '', 'GPT')
    chatMeta = dict(chatDao.getChat(userId, finalChatId))
    chatMeta["lastMessage"] = {
        "message": '',
        "_id": str(data.inserted_id) if hasattr(data, 'inserted_id')  else str(data.upserted_id) or 'lastMessage',
        "type": 'GPT'
    }
    chatMeta["_id"] = str(chatMeta["_id"])

    if data:
        return formatAndReturnResponse(chatMeta, status=status.HTTP_200_OK, isUI=isUI)

    return formatAndReturnResponse({'message': 'Failed to get last message' }, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

def getRedactedMetadata(userId, chatId, isUI):
    log.info('getRedactedMetadata')    
    chat = chatHistoryDao.getChatHistory(userId, chatId)
    if not chat:
        return formatAndReturnResponse({'message': 'No redacted meta data found for this message'}, status=status.HTTP_404_NOT_FOUND, isUI=isUI)
    
    data = json.loads(dumps(chat))
    if len(data):
        rMessages = data['messages'][::-1]
        messageWithRedactedMetadata = rMessages[0]
        data['messages'] = messageWithRedactedMetadata
        return formatAndReturnResponse(data, status=status.HTTP_200_OK, isUI=isUI)
    return formatAndReturnResponse({'message': 'No redacted meta data found for this message'}, status=status.HTTP_404_NOT_FOUND, isUI=isUI)


def getChatMessages(userId, chatId, page, pageSize, isUI):
    log.info('getChatMessages')
    chat = chatHistoryDao.getChatHistory(userId, chatId)

    if not chat:
        sharedChat = userChatShareDao.isChatIdSharedWithTheUser(chatId, userId)
        if not sharedChat:
            return formatAndReturnResponse({'message': 'Chat not found'}, status=status.HTTP_404_NOT_FOUND, isUI=isUI)
        chat = chatHistoryDao.getChatHistory(sharedChat["sharedBy"], chatId)
        if not chat:
            return formatAndReturnResponse({'message': 'Chat not found'}, status=status.HTTP_404_NOT_FOUND, isUI=isUI)

    data = json.loads(dumps(chat))
    if len(data):
        rMessages = data['messages'][::-1]
        paginator = Paginator(rMessages, pageSize)
        page_obj = paginator.get_page(page)
        messages = list()
        for message in page_obj:
            message['createdAt'] = convert_bson_datetime_to_string(message.get('createdAt'))
            message['modifiedAt'] = convert_bson_datetime_to_string(message.get('modifiedAt'))
            # Messages will be loaded in reverse order - scrolling up on UI will load older messages
            messages.insert(0, message)
        data['messages'] = messages
        meta = paginationMeta(paginator, page_obj, pageSize)

        return formatAndReturnResponse(data, status=status.HTTP_200_OK, isUI=isUI, pageInfo=meta)

    return formatAndReturnResponse({'message': 'No messages found for given chat id '}, status=status.HTTP_404_NOT_FOUND, isUI=isUI)

def deleteChatUserMessage(userId, chatId, messageId, isUI):
    log.info('deleteChatUserMessage')
    if not chatId or not messageId:
        return formatAndReturnResponse({'message': 'ChatId or messageId field is missing.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    chat = chatHistoryDao.deleteMessage(userId, chatId, messageId)
    if chat:
        return formatAndReturnResponse({ 'message': 'Successfully deleted chatId ' + chatId + ' and messageId ' + messageId}, status=status.HTTP_200_OK, isUI=isUI)
    return formatAndReturnResponse({'message': 'No chat found with chatId ' + chatId + ' or no user message found'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

def updateOrRegenerateChatUserMessage(userId, chatId, messageId, data, isUI, isRegenerate = False): # or regenerate
    log.info('updateOrRegenerateChatUserMessage')

    if not chatId:
        return formatAndReturnResponse({'message': 'ChatId or messageId field is missing.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    
    if not messageId and not isRegenerate:
        return formatAndReturnResponse({'message': 'Message is mandatory to update the message.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    chat = chatHistoryDao.deleteMessage(userId, chatId, messageId)
    if chat.modified_count > 0:
        return createOrUpdateChatMessage(userId, chatId, data, isUI, isRegenerate)
    return formatAndReturnResponse({'message': 'No chat found with chatId ' + chatId + ' or no user message found'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)


def getChatsByFilter(request, isUI):
    userId = request.GET.get('userId')
    userIds = userId.split(',') if userId else None
    chatId = request.GET.get('chatId')
    chatIds = chatId.split(',') if chatId else None
    messageType = request.GET.get('messageType')
    searchTerm = request.GET.get('searchTerm')

    messages = list()
    chats = list()
    if chatIds:
        chatIds = [ObjectId(chatId) for chatId in chatIds]
        chats = chatDao.getChatsFromIds(chatIds)
    elif not userIds:
        chats = chatDao.getChats()

    if userIds:
        chats = chatDao.getChats(userIds=userIds)

    chats = sorted(chats, key=lambda x: x['modifiedAt'])
    for chat in chats:
        chatId = str(chat["_id"])
        chatHistory = chatHistoryDao.getChatHistory(chat["userId"], chatId, includeDeleted=True, messageType=messageType, searchTerm=searchTerm)
        if not chatHistory:
            continue

        data = {
            'chatId': chatId,
            'userId': chat["userId"],
            'chat_name': chat["name"],
            'modifiedAt': chat["modifiedAt"],
            'messages': chatHistory['messages']
        }
        messages.insert(0, data)

    return formatAndReturnResponse(messages, status=status.HTTP_200_OK, isUI=isUI)

def getBookmarkedChatMessages(userId, page, pageSize, typeFilter, isUI):
    chatMeta = json.loads(dumps(chatDao.getBookmarkedChats(userId)))
    if chatMeta:
        bookmarkedChatMessages = json.loads(dumps(chatHistoryDao.getBookmarkedChatMessages(userId, typeFilter)))
        filteredChatMessages = list()
        for chat in chatMeta:
            for chatMessage in bookmarkedChatMessages:
                if chatMessage['chatId'] == chat['_id']['$oid']:
                    chatMessage['name'] = chat['name']
                    filteredChatMessages.append(chatMessage)

        for chatMessage in filteredChatMessages:
            for message in chatMessage['messages']:
                message['createdAt'] = convert_bson_datetime_to_string(message.get('createdAt'))
                message['modifiedAt'] = convert_bson_datetime_to_string(message.get('modifiedAt'))

        if filteredChatMessages and len(filteredChatMessages) > 0:# and type_filter_check:
            paginator = Paginator(filteredChatMessages, pageSize)
            page_obj = paginator.get_page(page)
            chatMetaList = list()
            for chat in page_obj:
                chatMetaList.append(chat)
            meta = paginationMeta(paginator, page_obj, pageSize)

            return formatAndReturnResponse(chatMetaList, status=status.HTTP_200_OK, isUI=isUI, pageInfo=meta)
    return formatAndReturnResponse({'message': 'No bookmarked chats found for given user id ' + str(userId) + ' and type ' + str(typeFilter)}, status=status.HTTP_404_NOT_FOUND, isUI=isUI)

def bookmarkChatMessage(userId, chatId, messageId, isUI):
    chatMeta = chatDao.bookmarkChat(userId, chatId) # 2nd time just reset "isBookmarked": True
    if chatMeta:
        chatHistory = chatHistoryDao.setBookmarkMessage(userId, chatId, messageId, True)
        if chatHistory:
            return formatAndReturnResponse({ 'message': 'Successfully Bookmarked chatId ' + chatId + ' and message with messageId ' + messageId}, status=status.HTTP_200_OK, isUI=isUI)
    return formatAndReturnResponse({'message': 'No chat found with chatId ' + chatId + ' or message with messageId ' + messageId}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

def removeBookmarkChatMessage(userId, chatId, messageId, isUI):
    chatHistory = chatHistoryDao.setBookmarkMessage(userId, chatId, messageId, False)
    if chatHistory:
        bookmarkedSome = chatHistoryDao.hasAnyBookmarkMessage(userId, chatId)
        if not bookmarkedSome:
            # if none of the chat messages are bookmarked then remove bookmark of chatMeta
            chatMeta = chatDao.removeBookmarkChat(userId, chatId)
            if chatMeta:
                return formatAndReturnResponse({ 'message': 'Removed Bookmarked chatId ' + chatId + ' message with messageId ' + messageId}, status=status.HTTP_200_OK, isUI=isUI)
            # Failed to remove bookmark from chatHistory messages
            return formatAndReturnResponse({'message': 'No chat found with chatId ' + chatId + ' or message with messageId ' + messageId}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
        # Message bookmark removed but chat bookmark kept as other messages is bookmarked
        return formatAndReturnResponse({'message': 'Successfully removed message with messageId ' + messageId}, status=status.HTTP_200_OK, isUI=isUI)
    return formatAndReturnResponse({'message': 'No chat found with chatId ' + chatId + ' or message with messageId ' + messageId}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    
def pinChat(userId, chatId, isUI):
    result = chatDao.pinChat(userId, chatId)
    if result.matched_count != 0:
        return formatAndReturnResponse({'message': 'Successfully pinned chat with id ' + chatId}, status=status.HTTP_200_OK, isUI=isUI)

    return formatAndReturnResponse({'message': 'No chat found with id ' + chatId}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

def unpinChat(userId, chatId, isUI):
    result = chatDao.unpinChat(userId, chatId)
    if result.matched_count != 0:
        return formatAndReturnResponse({'message': 'Successfully unpinned chat with id ' + chatId}, status=status.HTTP_200_OK, isUI=isUI)

    return formatAndReturnResponse({'message': 'No chat found with id ' + chatId}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

def shareChats(userId, chatId, sharedWith, name, isUI):
    log.info('shareChats')
    if not chatId or not userId:
        return formatAndReturnResponse({'message': 'ChatId or userIds or sharedWith fields are mandatory.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    sharedWithEveryOne = len(sharedWith) == 0
    sharedWith = sharedWith if len([sharedWith]) else [user.id for user in list(User.objects.exclude(id=userId))]


    # TODO: With this we can even detect what all user id's are invalid but we need to think before returning on the UI because it can open the security threat of knowing valid customer ids.
        # existingUserIds = User.objects.filter(id__in=sharedWith).values_list('id', flat=True)
        # existingUserIdsSet = set(existingUserIds)

        # nonExistingUserIds = list(set(existingUserIds) - existingUserIdsSet)
        # existingUserIds = list(existingUserIds)
    # ##
    # existingUserIds = User.objects.filter(id__in=sharedWith).values_list('id', flat=True)
    existingUserIds = sharedWith
    allExist = set(sharedWith) == set(existingUserIds)

    if not allExist:
        return formatAndReturnResponse({'message': 'Few user Ids are invalid. Failed to share'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    response = userChatShareDao.shareChat(sharedBy=userId, chatId=chatId, sharedWith=sharedWith, name=name)
    if response.acknowledged:
        if not sharedWithEveryOne:
            sharedByUser = getOrNone(model=User, id=userId)
            for chatSharedWithUserId in sharedWith:
                sharedWithUser = getOrNone(model=User, id=chatSharedWithUserId)
                shareUrl = f"{settings.CONFIG_UI_HOST}?id={chatId}"
                def templateReplacer(template):
                    template = template.replace("{{username}}", sharedWithUser.firstName)
                    template = template.replace("{{sharedBy}}", sharedByUser.firstName)
                    template = template.replace("{{shareUrl}}", shareUrl)
                    return template

                EmailClient.sendEmailWithTemplate(
                    [sharedWithUser.email],
                    f"{sharedByUser.firstName} shared chat with you",
                    f"{settings.BASE_DIR}/templates/emails/shareChat.html",
                    templateReplacer
                )
        return formatAndReturnResponse({'message': 'Chat Shared Successfully.'}, status=status.HTTP_200_OK, isUI=isUI)
    else:
        return formatAndReturnResponse({'message': 'Failed to share the chat. Please try again'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, isUI=isUI)

def getSharedChats(userId, isUI):
    log.info('getSharedChats')
    if not userId:
        return formatAndReturnResponse({'message': 'ChatId or userIds fields are mandatory.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    chatIdsSharedByUser = userChatShareDao.getChatIdsSharedByUser(sharedBy=userId)
    chatIdsSharedWithUser = userChatShareDao.getChatIdsSharedWithUser(userId=userId)
    
    return formatAndReturnResponse({
        'chatIdsSharedByUser': chatIdsSharedByUser,
        'chatIdsSharedWithUser': chatIdsSharedWithUser
    }, status=status.HTTP_200_OK, isUI=isUI)

def revokeSharedChat(chatId, sharedBy, userIds, isUI):
    log.info('revokeSharedChat')
    if not chatId:
        return formatAndReturnResponse({'message': 'ChatId field is mandatory.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    response = userChatShareDao.revokeSharedChatAccess(chatId, sharedBy, userIds)
    if response.deleted_count == 1 or response.modified_count == 1:
        return formatAndReturnResponse({'message': 'Revoked Access Successfully.'}, status=status.HTTP_200_OK, isUI=isUI)
    else:
        return formatAndReturnResponse({'message': 'Failed to revoke access. Please try again'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, isUI=isUI)

def selfRemoveFromSharedList(chatId, sharedBy, userId, isUI):
    log.info('selfRemoveFromSharedList')
    if not chatId or not sharedBy:
        return formatAndReturnResponse({'message': 'ChatId and sharedBy field is mandatory.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    response = userChatShareDao.excludeFromSharing(chatId,sharedBy, userId)
    if response.modified_count == 1:
        return formatAndReturnResponse({'message': 'Revoked Access Successfully.'}, status=status.HTTP_200_OK, isUI=isUI)
    else:
        return formatAndReturnResponse({'message': 'Failed to revoke access. Please try again'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, isUI=isUI)
    

def importChat(userChatSharingId, oldChatId, userId, isUI):
    log.info('importChat')
    if not oldChatId or not userId:
        return formatAndReturnResponse({'message': 'ChatId and UserId fields are mandatory.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    
    if not userChatShareDao.isChatIdSharedWithTheUser(oldChatId, userId):
        return formatAndReturnResponse({'message': 'Unauthorized access.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    sourceChatHistory = userChatShareDao.getSharedChatHistory(userChatSharingId, oldChatId)
    chatCreated = chatDao.importChat(oldChatId, userId, sourceChatHistory['name'])

    if not chatCreated:
        return formatAndReturnResponse({'message': 'Chat Not found'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    if not chatCreated.acknowledged:
        return formatAndReturnResponse({'message': 'Failed to import chat'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, isUI=isUI)
    
    newChatId = str(chatCreated.inserted_id)

    chatHistoryImported = chatHistoryDao.importChatHistory(newChatId, userId, sourceChatHistory)
    if not chatHistoryImported.acknowledged:
        chatDao.deleteChat(newChatId, userId) #TODO handle retry later at some point
        return formatAndReturnResponse({'message': 'Failed to import chat'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, isUI=isUI)
    
    return formatAndReturnResponse({'message': 'Imported Chat Successfully.', "chatId": newChatId }, status=status.HTTP_200_OK, isUI=isUI)

def updateChatName(userId, chatId, name, isUI):
    result = chatDao.updateChatName(userId, chatId, name)
    if result.matched_count != 0:
        return formatAndReturnResponse({'message': 'Successfully updated name for chat with id ' + chatId},
                                       status=status.HTTP_200_OK, isUI=isUI)

    return formatAndReturnResponse({'message': 'No chat found with id ' + chatId}, status=status.HTTP_400_BAD_REQUEST,
                                   isUI=isUI)

def createLlmModel(data, isUI):
    modelType = data['modelType'] # name on UI
    secretKey = data['secretKey'] # API KEY
    version = data['modelVersion'] # Model version on UI
    apiURL = data['apiURL'] # API URL for LLAMA model
    if not isValidKey(modelType, secretKey):
            return formatAndReturnResponse({'message': 'Invalid Secret key entered.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    if modelType not in TYPE_OF_AI:
            return formatAndReturnResponse({'message': 'Currently we only support 2 model types: GPT and LLAMA'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    secretKey = crypto.encrypt(secretKey.encode('utf-8')).decode('utf-8')
    response = llmModelDao.createLlmModel(modelType, version, secretKey, apiURL)
    if not response:
            if response is None or not response.acknowledged:
                formatAndReturnResponse({'message': 'Failed to create details.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    return formatAndReturnResponse({"message": 'Model details saved succesfully', "id": str(response.inserted_id)}, status=status.HTTP_200_OK, isUI=isUI)

def getLlmModels(isUI):
    models = json.loads(dumps(llmModelDao.getLlmModels()))
    if len(models):
        for model in models:
            model['secretKey'] = crypto.decrypt(model['secretKey'].encode('utf-8')).decode('utf-8')

        return formatAndReturnResponse(models, status=status.HTTP_200_OK, isUI=isUI)
    return formatAndReturnResponse([], status=status.HTTP_200_OK, isUI=isUI)

def updateLlmModel(data, modelId, isUI):
    if not modelId:
        formatAndReturnResponse({'message': 'Model id field is compulsory'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    model = llmModelDao.getLlmModel(modelId)

    if not model:
        formatAndReturnResponse({'message': 'Model details not found'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    isDefault = data.get('isDefault', model["isDefault"])
    modelType = data.get('modelType', model["type"])
    secretKey = data.get('secretKey', crypto.decrypt(model["secretKey"].encode('utf-8')).decode('utf-8'))
    version = data.get('modelVersion', model["modelVersion"])
    apiURL = data.get('apiURL', model["apiURL"])

    if not isValidKey(modelType, secretKey):
        return formatAndReturnResponse({'message': 'Invalid Secret key entered.'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    
    if modelType not in TYPE_OF_AI:
            return formatAndReturnResponse({'message': 'Currently we only support 2 model types: GPT and LLAMA'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    secretKey = crypto.encrypt(secretKey.encode('utf-8')).decode('utf-8')
    
    response = llmModelDao.updateLlmModel(modelId, modelType, version, secretKey, apiURL, isDefault)
    if not response:
        if response is None or not response.acknowledged:
            formatAndReturnResponse({'message': 'Failed to save details.'}, status=status.HTTP_400_BAD_REQUEST,
                                    isUI=isUI)  
    return formatAndReturnResponse({"message": 'Model details saved succesfully'}, status=status.HTTP_200_OK, isUI=isUI)

def deleteModel(modelId, isUI):
    if not modelId:
        formatAndReturnResponse({'message': 'Model id field is compulsory'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)

    model = llmModelDao.getLlmModel(modelId)

    if not model:
        formatAndReturnResponse({'message': 'Model details not found'}, status=status.HTTP_400_BAD_REQUEST, isUI=isUI)
    
    response = llmModelDao.deleteModel(modelId)
    if not response:
            return formatAndReturnResponse({'message': 'Default cannot be deleted. Please make another model as default and delete this one.'}, status=status.HTTP_400_BAD_REQUEST,
                                    isUI=isUI)  
    return formatAndReturnResponse({"message": 'Model deleted succesfully'}, status=status.HTTP_200_OK, isUI=isUI)

def isValidKey(vendor, key):
    if vendor == 'GPT':
        openai.api_key = key
        try:
            openai.Model.list()
        except openai.error.AuthenticationError:
            return False

    return True
