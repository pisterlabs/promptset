# -------------------------------------------------------------------------------------------------
# This file contains the JSON endpoints for AI Chat posts, handling the CRUD operations with the db 
#
from fastapi import APIRouter, HTTPException, Path, Depends, status

from app.api import crud
from typing import List
from app.api.users import get_current_active_user, user_has_role
from app.api.user_action import UserAction, UserActionLevel
from app.api.models import UserInDB, ChatbotDB, AiChatDB, AiChatTask, AiChatCreate, AiChatCreateResponse
from app.api.models import ProjectDB, TagDB, basicTextPayload

from app.config import log, get_settings
import json

import asyncio
import openai

from app.worker import celery_app

# ---------------------------------------------------------------------------------------

# Celery specific:
from app.worker import OpenAI_Comm
from celery.result import AsyncResult

# ---------------------------------------------------------------------------------------

router = APIRouter()

openai.api_key = get_settings().OPENAI_API_KEY

# ----------------------------------------------------------------------------------------------
# declare a POST endpoint on the root 
@router.post("/", response_model=AiChatCreateResponse, status_code=201)
async def create_aiChatExchange(payload: AiChatCreate, 
                                current_user: UserInDB = Depends(get_current_active_user)) -> AiChatCreateResponse:
    
    chatbot: ChatbotDB = await crud.get_Chatbot(payload.chatbotid)
    if not chatbot:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_POST_NEW_AICHAT'), 
                                       f"chatbot {payload.chatbotid}, not found" )
        raise HTTPException(status_code=404, detail="chatbot not found")
    
    proj: ProjectDB = await crud.get_project(chatbot.projectid)
    if not proj:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_POST_NEW_AICHAT'), 
                                       f"Project {chatbot.projectid}, not found" )
        raise HTTPException(status_code=404, detail="AIChat Project not found")
        
    tag: TagDB = await crud.get_tag( proj.tagid )
    if not tag:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_POST_NEW_AICHAT'), 
                                       f"Project Tag {proj.tagid}, not found" )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AIChat Project Tag not found")
    
    weAreAllowed = crud.user_has_project_access( current_user, proj, tag )
    if not weAreAllowed:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('WARNING'),
                                       UserAction.index('FAILED_POST_NEW_AICHAT'), 
                                       f"Project {proj.projectid}, '{proj.name}', not authorized" )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, 
                            detail="Not Authorized to create AI Chats for project.")
    
        
    # first create a db entry for this new aichat 
    aichatid = await crud.post_aiChat(payload.prompt, chatbot, current_user)
    #
    log.info(f"create_aiChatExchange: aichatid '{aichatid}'")
    #
    await crud.rememberUserAction( current_user.userid, 
                                   UserActionLevel.index('NORMAL'),
                                   UserAction.index('POST_NEW_AICHAT'), 
                                   f"aichatid {aichatid}" )
    #
    aichat: AiChatDB = await crud.get_aiChat( aichatid )
    #
    # sends a message to Celery that runs a Task to do this communication:
    aichat_task = AiChatTask( prePrompt = aichat.prePrompt, 
                              prompt = aichat.prompt, 
                              reply = aichat.reply, 
                              model = aichat.model,
                              status = aichat.status,
                              taskid = aichat.taskid,
                              aichatid = aichat.aichatid )
    #
    log.info(f"create_aiChatExchange: created aichat_task, launching task...")
    #
    task = OpenAI_Comm.delay( aichat_task ) 
    #
    log.info(f"create_aiChatExchange: ...back from task launch, taskid is: {task.id}")
    #
    # update in db with 'inuse' status, which signals the get endpoints to check for the task finishing:
    aichat.status = 'inuse'
    aichat.taskid = task.id
    #
    retVal = await crud.put_aichat( aichat )
    
    await crud.rememberUserAction( aichat.userid, 
                                   UserActionLevel.index('NORMAL'),
                                   UserAction.index('UPDATE_AICHAT'), 
                                   f"aiChat {retVal} waiting on Celery as task {task.id}" )
    
    return { "aichatid": aichatid }

# ----------------------------------------------------------------------------------------------
# Note: id's type is validated as greater than 0  
@router.get("/{id}", response_model=AiChatDB)
async def read_aichatExchange(id: int = Path(..., gt=0),
                              current_user: UserInDB = Depends(get_current_active_user)) -> AiChatDB:
    
    aichat = await crud.get_aiChat(id)
    if aichat is None:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_GET_AICHAT'), 
                                       f"AIChat {id}, not found" )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AIChat not found")
    
    proj, tag = await crud.get_project_and_tag(aichat.projectid)
    if proj is None:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_GET_AICHAT'), 
                                       f"Project {aichat.projectid}, not found" )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    #
    if not tag:
        await crud.rememberUserAction( current_user.userid,  
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_GET_AICHAT'), 
                                       f"Project {id}, '{proj.name}', Tag not found" )
        raise HTTPException(status_code=500, detail="Project Tag not found")
    
    weAreAllowed = crud.user_has_project_access( current_user, proj, tag )
    if not weAreAllowed:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('WARNING'),
                                       UserAction.index('FAILED_GET_AICHAT'), 
                                       "Not Authorized" )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not Authorized to access project.")
        
    await crud.rememberUserAction( current_user.userid, 
                                   UserActionLevel.index('NORMAL'),
                                   UserAction.index('GET_AICHAT'), 
                                   f"AIChat {aichat.aichatid}" )
    
    return aichat

# ----------------------------------------------------------------------------------------------
# The response_model is a List with a AiChatDB subtype. See import of List top of file. 
# Returns a list of the projects the user has access.
@router.get("/project/{projectid}", response_model=List[AiChatDB])
async def read_all_project_aichats(projectid: int = Path(..., gt=0),
                                   current_user: UserInDB = Depends(get_current_active_user)) -> List[AiChatDB]:
    
    proj, tag = await crud.get_project_and_tag(projectid)
    if proj is None:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_GET_AICHATLIST'), 
                                       f"Project {projectid}, not found" )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    #
    if not tag:
        await crud.rememberUserAction( current_user.userid,  
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_GET_AICHATLIST'), 
                                       f"Project {id}, '{proj.name}', Tag not found" )
        raise HTTPException(status_code=500, detail="Project Tag not found")
    
    weAreAllowed = crud.user_has_project_access( current_user, proj, tag )
    if not weAreAllowed:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('WARNING'),
                                       UserAction.index('FAILED_GET_AICHATLIST'), 
                                       "Not Authorized" )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not Authorized to access project.")
    
    # get all the ai chats for this project:
    aichatsList = await crud.get_all_project_aiChats(projectid)
    
    await crud.rememberUserAction( current_user.userid, 
                                   UserActionLevel.index('NORMAL'),
                                   UserAction.index('GET_AICHATLIST'), 
                                   f"AIChat for Project {projectid}, {proj.title}" )
    
    return aichatsList

# ----------------------------------------------------------------------------------------------
# Note: id's type is validated as greater than 0  
@router.put("/{aichatid}", response_model=int)
async def update_aiChatExchange(payload: basicTextPayload,       # a new question
                                aichatid: int = Path(..., gt=0), # the aichatid
                                current_user: UserInDB = Depends(get_current_active_user)) -> int:

    log.info(f"update_aiChatExchange: here!")
    
    aichat = await crud.get_aiChat(aichatid)
    if aichat is None:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_UPDATE_AICHAT'), 
                                       f"AIChat {id} not found" )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AIChat not found")
    
    proj: ProjectDB = await crud.get_project(aichat.projectid)
    if proj is None:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_UPDATE_AICHAT'), 
                                       f"Project {aichat.projectid}, not found" )
        raise HTTPException(status_code=404, detail="Project not found")
        
    tag = await crud.get_tag( proj.tagid )
    if not tag:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_UPDATE_AICHAT'), 
                                       f"Project Tag {proj.tagid}, not found" )
        raise HTTPException(status_code=500, detail="Project Tag not found")
        
    weAreAllowed = crud.user_has_project_access( current_user, proj, tag )
    if not weAreAllowed:
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('WARNING'),
                                       UserAction.index('FAILED_UPDATE_AICHAT'), 
                                       "Not Authorized" )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not Authorized to access project.")
        
    log.info(f"update_aiChatExchange: aichat.status is {aichat.status}")
    
    # do not allow updates when a Celery task is processing:
    if aichat.status == 'inuse':
        await crud.rememberUserAction( current_user.userid, 
                                       UserActionLevel.index('SITEBUG'),
                                       UserAction.index('FAILED_UPDATE_AICHAT'), 
                                       f"AIChat {id} still processing last request" )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="AIChat still processing last request")
    
    # new prompt is the prior conversation with our new question appended:
    aichat.prompt = aichat.prompt + "<br><br>" + aichat.reply + "<br><br>" + payload.text
    aichat.reply = ''
    
    #
    # sends a message to Celery that runs a Task to do this OpenAI communication:
    aichat_task = AiChatTask( prePrompt = aichat.prePrompt, 
                              prompt = aichat.prompt, 
                              reply = aichat.reply, 
                              model = aichat.model,
                              status = aichat.status,
                              taskid = aichat.taskid,
                              aichatid = aichat.aichatid )
    #
    log.info(f"update_aiChatExchange: created aichat_task, launching update task...")
    #
    task = OpenAI_Comm.delay( aichat_task ) 
    #
    log.info(f"update_aiChatExchange: ...back from update task launch, taskid is: {task.id}" )
    #
    aichat.status = 'inuse'
    aichat.taskid = task.id
    #
    # remember this status and taskid
    retVal = await crud.put_aichat( aichat )
    
    await crud.rememberUserAction( aichat.userid, 
                                   UserActionLevel.index('NORMAL'),
                                   UserAction.index('UPDATE_AICHAT'), 
                                   f"aiChat {retVal} waiting on Celery as task {task.id}" )
    
    return retVal
