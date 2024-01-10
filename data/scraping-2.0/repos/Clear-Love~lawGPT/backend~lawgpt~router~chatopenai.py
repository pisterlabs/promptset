import json
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException
import openai
from lawgpt.log import get_logger
from lawgpt.models.dbModels import User
from lawgpt.models.reqModels import ChatCompletionRequest, ChatMessage, updateTitleRequest
from lawgpt.models.respModels import ChatCompletionResponse, ChatCompletionResponseChoice
from sse_starlette.sse import EventSourceResponse
from lawgpt.services.chatService import ChatService
from lawgpt.services.vecDBService import vecDBService
from langchain.prompts import PromptTemplate
from lawgpt.config import settings
from lawgpt.utils.user_manager import current_active_user
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from lawgpt.models.respModels import GenTitleRequest
from lawgpt.models.respModels import SearchResponse

router = APIRouter()
logger = get_logger(__name__)
_service = ChatService()
vecDB = vecDBService()
os.environ['OPENAI_API_KEY'] = settings.API_KEY
openai.api_key = settings.API_KEY

TEMPLATE = """
            {docs}
            根据以上法律条文作为回答的依据，并在回答中引用使用到的法律条文，如果法律条文未与问题内容相关则无视该法律条文，问题是：
            “{query}”
        """
resp_prompt = PromptTemplate(
    input_variables=["query", "docs"],
    template=TEMPLATE
)

gen_prompt = PromptTemplate(
    input_variables=["history"],
    template="{history}, 总结一下上述对话，返回不超过15个字的标题，尽量简短"
)


@router.post("/chat/{conversation_id}", response_model=ChatCompletionResponse)
async def create_chat_completion(conversation_id: str,
                                 request: ChatCompletionRequest,
                                 current_user: User = Depends(current_active_user)
                                 ):
    conversation = await _service.get_conversation_by_id(conversation_id=conversation_id)
    if not conversation or current_user.id != conversation.user_id or not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=502, detail="Invalid request")
    query = request.messages[-1].content
    docs = vecDB.get_knowledge(query, top_k=request.top_k)
    query = str(resp_prompt.format_prompt(query=query, docs=docs))
    request.messages[-1].content = query
    messages = []
    messages.append(ChatMessage(role="system", content="你是法律问答助手lawGPT，你可以根据提供的知识回答法律问题，你需要在回答中扮演一名法律顾问").model_dump())
    for msg in request.messages:
        messages.append(msg.model_dump())
    logger.info(f'查询到的topk文档{docs}')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        top_p=request.top_p,
        temperature=request.temperature,
        stream=request.stream
    )
    if request.stream:
        async def generate():
            content = []
            for chunk in response:
                yield json.dumps(chunk)
                if chunk['choices'][0]['finish_reason'] != 'stop':
                    content.append(chunk['choices'][0]['delta']['content'])
            yield '[DONE]'
            content = ''.join(content)
            logger.info(f'完整回答：{content}')
        return EventSourceResponse(generate(), media_type="text/event-stream")

    content = response['choices'][0]['message']['content']
    content += f"\n{docs}"
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop"
    )
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

@router.post('/conv/gen-title/{conversation_id}', response_model=GenTitleRequest)
async def gen_title(conversation_id: str,
                    request: List[ChatMessage],
                    current_user: User = Depends(current_active_user)):
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=gen_prompt,
        verbose=True,
    )
    history = []
    for i in range(0, len(request), 2):
        if request[i].role == "user" and request[i+1].role == "assistant":
            history.append([request[i].content,
                            request[i+1].content])
    title = str(chatgpt_chain.run(history=history))
    title = title.removeprefix('：\n\n')
    await _service.set_title(conversation_id, current_user.id, title)
    return GenTitleRequest(title=title)


@router.post("/search/{conversation_id}", response_model=SearchResponse)
async def chat_search(conversation_id: str,
                    request: ChatCompletionRequest,
                    current_user: User = Depends(current_active_user)
                    ):
    conversation = await _service.get_conversation_by_id(conversation_id=conversation_id)
    if not conversation or current_user.id != conversation.user_id or not request.messages or request.messages[-1].role != "user":
        raise HTTPException(status_code=502, detail="Invalid request")
    query = request.messages[-1].content
    docs = vecDB.get_knowledge(query, top_k=request.top_k)
    query = str(resp_prompt.format_prompt(query=query, docs=docs))
    return SearchResponse(docs=docs)