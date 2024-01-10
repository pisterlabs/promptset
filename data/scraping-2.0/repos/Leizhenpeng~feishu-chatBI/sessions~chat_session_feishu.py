from configs import chat_save_file_path_feishu
import shutil
import queue
import repository.feishu as feishu_repo
from codeinterpreterapi import CodeInterpreterSession, File
from custom_codeinterpreter_session import CustomCodeInterpreterSession
import models.feishu_api as models
import json
import os
from typing import Any, Dict, List, Optional
from langchain.schema import LLMResult
from langchain.callbacks.base import  BaseCallbackHandler
from uuid import UUID
from utils.file_util import get_feishu_file_type
from langchain.schema import (
    LLMResult,
)
from models.feishu_api import FeishuChatSessionKey,FeishuAppInfo
from custom_codeinterpreter_session import InstantMessageCallbackHandler
import asyncio
from utils.feishu_util import renew_tenant_access_token,get_tenant_access_token
import time
from configs import DEFAULT_FEISHU_APP_INFO


class ChatSessionFeishu(object):
    TAG = "ChatSessionFeishu"

    def __init__(self,session_key:FeishuChatSessionKey, app_info:FeishuAppInfo=DEFAULT_FEISHU_APP_INFO):
        self.app_info=app_info
        self.verbose=True
        self.running_lock=asyncio.Lock()
        self.async_message_queue = asyncio.Queue()
        self.session_key=session_key
        self.chat_id = session_key.chat_id
        self.input_files = []
        self.save_file_dir = f'{chat_save_file_path_feishu}{self.chat_id}/'
        self.output_file_save_dir=f'{chat_save_file_path_feishu}{self.chat_id}/output_files/'
        self.send_files = []
        self.tenant_access_token=None
        self.tenant_access_token_expire=0
        self._init_tenant_access_token()
        for item in [self.save_file_dir,self.output_file_save_dir]:
            if not os.path.exists(item):
                os.makedirs(item)
    
    def _init_tenant_access_token(self):
        (self.tenant_access_token,self.tenant_access_token_expire) = get_tenant_access_token(app_id=self.app_info.app_id, app_secret=self.app_info.app_secret)
        if self.tenant_access_token is None and self.tenant_access_token_expire == 0:
            raise ValueError
    
    def renew_tenant_access_token(self):
        if int(time.time()) >= self.tenant_access_token_expire:
            self._init_tenant_access_token()

    def to_log(self):
        logs = f"{self.TAG} obj_address={id(self)}, session_key={self.session_key}\n chat_id={self.chat_id}\n input_files={self.input_files}, send_files={self.send_files}"
        if self.verbose:
            print(logs)
        return logs

    async def astart(self) -> None:
        instant_msg_callback=InstantMessageCallbackHandler()
        instant_msg_callback.callback=self.on_message_generated
        print("_astart to get ci session")
        async with CustomCodeInterpreterSession(openai_api_key=self.app_info.openai_api_key,chat_session_callback=instant_msg_callback,agent_executor_callback_func=self.on_message_generated ) as session:
            session.output_file_save_path=self.output_file_save_dir
            session.on_file_saved_callback=self.on_output_file_saved
            self.code_interpreter_session: CustomCodeInterpreterSession = session
    

    async def __aenter__(self) -> "ChatSessionFeishu":
        await self.astart()
        return self
    
    async def astop(self) -> None:
        await self.code_interpreter_session.astop()


    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        # await self.astop()
        pass


    
    def on_file_upload(self, file_saved_path: str):
        self.input_files.append(file_saved_path)
        self.to_log()
        return
    
    def on_output_file_saved(self, saved_file_path: str):
        if self.verbose:
            print(f"{self.TAG} on_output_file_saved={saved_file_path}")
        if saved_file_path in self.send_files:
            return
        self.send_files.append(saved_file_path)
        _ext = os.path.splitext(os.path.basename(saved_file_path))[-1].lower()
        if _ext=='.png':
            image_key=feishu_repo.upload_image(
                image_path=saved_file_path,
                upload_body=models.UploadImageBody(
                    image_type='message'
                ),
                tenant_access_token=self.tenant_access_token
            )
            feishu_repo.send_message(
                receive_id_type="chat_id",
                body_data=models.FeishuSendMessageBody(
                    receive_id=self.chat_id,
                    msg_type='image',
                    content=json.dumps({"image_key":image_key})
                ),
                tenant_access_token=self.tenant_access_token
            )
        else:
            file_key=feishu_repo.upload_file(
                file_path=saved_file_path,
                body=models.UploadFileBody(
                    file_type=get_feishu_file_type(saved_file_path),
                    file_name=os.path.basename(saved_file_path)
                ),
                tenant_access_token=self.tenant_access_token
            )
            feishu_repo.send_message(
                receive_id_type="chat_id",
                body_data=models.FeishuSendMessageBody(
                    receive_id=self.chat_id,
                    msg_type='file',
                    content=json.dumps({"file_key":file_key})
                ),
                tenant_access_token=self.tenant_access_token
            )
        return

    def on_message_generated(self, text_generated: str):
        feishu_repo.send_message(
            receive_id_type="chat_id",
            body_data=models.FeishuSendMessageBody(
                receive_id=self.chat_id,
                msg_type='text',
                content=json.dumps({"text":text_generated})
            ),
            tenant_access_token=self.tenant_access_token
        )
    
    def on_message_state_update(self, message_id, text):
        feishu_repo.reply_message(
            message_id=message_id, 
            data= models.FeishuReplyMessageData(
                content=json.dumps({"text":text}),
                msg_type='text'
            ),
            tenant_access_token=self.tenant_access_token
        )
    
    async def on_new_text_message_entered(self, msg:str):
        if self.verbose:
            print(f"{self.TAG} on_new_text_message_entered({msg})")
        
        await self.async_message_queue.put(msg)
        while True:
            # 试图获取锁
            async with self.running_lock:
                if not self.async_message_queue.empty():
                    msg = await self.async_message_queue.get()
                    await self.chat(input=msg)
                    break
                else:
                    break

    async def chat(self,input:str) -> str:
        user_request=input
        files = [
            File.from_path(item) for item in self.input_files
        ]
        # generate the response
        response = await self.code_interpreter_session.agenerate_response(
            user_request, files=files
        )
        if response.content=="Sorry, something went while generating your response.Please try again or restart the session.":
            self.on_message_generated("对不起，我的处理出现了些问题，请稍后重试或者联系管理开发人员。")
        print("AI: ", response.content)
        return response.content

    def __del__(self):
        print(f"{self.TAG} __del__({self.chat_id})")
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.code_interpreter_session.astop())
            shutil.rmtree(self.save_file_dir)
        except Exception as e:
            print(e)


class MyCustomSyncHandler(BaseCallbackHandler):
    last_prompt = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.last_prompt = prompts
        pass
    
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    