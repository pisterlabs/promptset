from concurrent import futures
import logging

import grpc
import ask_pb2 
import ask_pb2_grpc
import dialogue_pb2
import dialogue_pb2_grpc
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    prompt
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Any
from main import Aifred
from prompt.template_maker import TemplateMaker

class Asker(ask_pb2_grpc.AskerServicer):
    def Ask(self, request, context):
        result = Aifred().process(request.question)
        return ask_pb2.AskReply(**result)

class Communicator(dialogue_pb2_grpc.CommunicatorServicer):

    def searchContent(self, request, context):
        result = Aifred().searchContent(request.text)
        return dialogue_pb2.Content(content=result)


    def askStreamReply(self
                       , request: dialogue_pb2.Conversation
                       , context) -> dialogue_pb2.Message:
        ''' 질문에 대한 응답을 스트리밍으로 전달하는 메소드 '''

        print("request : ", request)

        # 1. 참고 내용을 가져온다.
        contentMsg = "" #str(doc)
        contentList = request.content
        if (len(contentList) > 0):
            # 시간으로 내림차순 정렬하고 1번째 항목을 가져온다.
            sorted_list = sorted(contentList, key=lambda x: x.time, reverse=True)
            contentMsg = sorted_list[0].content

        # 2. 질문을 가져온다.
        prompt = request.message.text

        # 사용자에게 전달할 결과(Iterator)
        resultIter = None

        # type에 따른 분기처리
        #   (1: 사용자의 질문, 2: 시스템의 답변, 3: 시스템의 질문, 4: 사용자의 답변 )
        if "1" == request.message.type:
            chat_result = None
            # 질문에 대한 추가적인 정보가 필요한지 확인한다.
            if len(contentList) > 0:
                prompt = TemplateMaker.makeTemplateText('CONFIRM_QUESTION_01', [contentMsg, prompt])

                chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
                sys = SystemMessage(content="")
                msg = HumanMessage(content=prompt)
                chat_result = chat([sys, msg])
            
            # 추가적인 정보가 필요하다면 -> 추가적인 정보를 요청한다.
            if chat_result is not None and "no message" not in chat_result.content:
                for char in iter(chat_result.content):
                    yield dialogue_pb2.Message(text=char, type="3")

            # 추가적인 정보가 필요없다면 -> 답변을 생성한다.
            else:
                prompt = TemplateMaker.makeTemplateText('ANSWER_02', [contentMsg, prompt])

                chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', temperature=0.9)
                sys = SystemMessage(content="")
                msg = HumanMessage(content=prompt)
                resultIter = chat.stream([sys, msg])

        elif "2" == request.message.type:
            pass
        elif "3" == request.message.type:
            pass
        elif "4" == request.message.type:
            question = ""
            # 시간으로 내림차순 정렬하고 - type이 1인 첫번째 항목을 가져온다.
            if len(request.messageHistory) > 0:
                sorted_list = sorted(request.messageHistory, key=lambda x: x.time, reverse=True)
                for msg in sorted_list:
                    if "1" == msg.type:
                        question = msg.text
                        break

            # contentMsg=약관, question=질문(이전질문), prompt=참고사항(사용자의 답변)
            prompt = TemplateMaker.makeTemplateText('ANSWER_01', [contentMsg, question, prompt])

            chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', temperature=0.9)
            sys = SystemMessage(content="")
            msg = HumanMessage(content=prompt)
            resultIter = chat.stream([sys, msg])
            pass
        else:
            chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', temperature=0.9)
            sys = SystemMessage(content=contentMsg)
            msg = HumanMessage(content=prompt)
            resultIter = chat.stream([sys, msg])
            pass

        # 답변을 전달한다.
        for result in resultIter:
            yield dialogue_pb2.Message(text=result.content, type="2")



def serve():
    port = os.environ.get('SERVER_PORT')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # add service
    ask_pb2_grpc.add_AskerServicer_to_server(Asker(), server)
    dialogue_pb2_grpc.add_CommunicatorServicer_to_server(Communicator(), server)

    # start server
    server.add_insecure_port("[::]:" + port) # 인증없이 사용할 수 있도록 설정, 운영환경에서는 add_secure_port를 사용해야 함
    server.start()
    print(f"Server started, listening {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
