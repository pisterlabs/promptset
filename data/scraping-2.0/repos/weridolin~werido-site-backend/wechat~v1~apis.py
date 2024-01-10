from rest_framework.viewsets import ModelViewSet
from rest_framework import parsers
from wechat.models import WechatMessage
from wechat.v1.seriazliers import WechatMessageSerializer
from rest_framework_xml.parsers import XMLParser
from rest_framework_xml.renderers import XMLRenderer
import hashlib
import os
from utils.http_ import HTTPResponse
from rest_framework import status
from rest_framework import renderers,parsers
from rest_framework.response import Response
# from django.http import request
from wechat.v1.msg_handlers import text_msg_handler,DEFAULT_REPLY_CONTENT,TEXT_REPLY_XML_TEMPLATE,WELCOME_CONETENT,get_chatGPT_response
import openai


class CheckTokenRender(renderers.JSONRenderer):

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return data.encode()

class WeChatMessageReceiveParser(XMLParser):

    media_type="text/xml" #微信的xml request content-type

class PublicCountMessageApis(ModelViewSet):

    model = WechatMessage
    parser_classes=[WeChatMessageReceiveParser]
    serializer_class =WechatMessageSerializer
    renderer_classes=[renderers.JSONRenderer]
    queryset = WechatMessage

    def create(self, request, *args, **kwargs):
        """
            {
                'ToUserName': 'gh_fbf9348bd3ad', 
                'FromUserName': 'odZCO6Wh2smxI0ZuT33CnpEvGD1E', 
                'CreateTime': 1673250920, 
                'MsgType': 'text', 
                'Content': 111, 
                'MsgId': 23955193308095025
            }
        """
        msg_type = request.data.get("MsgType")
        serializer = WechatMessageSerializer(
            data= {
                "msg_id":request.data.get("MsgId"),
                "msg_to":request.data.get("ToUserName"),
                "msg_from":request.data.get("FromUserName"),
                "create_time":request.data.get("CreateTime"),
                "msg_type":request.data.get("MsgType"),
            }
        )
        reply=""
        if msg_type =="event":
            event_type = request.data.get("Event")
            assert event_type is not None,"事件类型不能为空"
            if event_type.lower() =="subscribe":
                reply = TEXT_REPLY_XML_TEMPLATE.format(
                    to=request.data.get("FromUserName"),
                    from_=request.data.get("ToUserName"),
                    content=WELCOME_CONETENT
                )
            if event_type.lower() =="unsubscribe":
                ...
            if event_type.lower() == "scan":
                print(">>> user scan")
            if event_type.lower() == "click":
                ...
        elif msg_type=="text":
            serializer.is_valid(raise_exception=True)
            reply = text_msg_handler(
                to=request.data.get("FromUserName"),
                from_=request.data.get("ToUserName"),
                content=request.data.get("Content")
            )
            print(">>> get wechat message",reply)
        else:
            reply = TEXT_REPLY_XML_TEMPLATE.format(
                to=request.data.get("FromUserName"),
                from_=request.data.get("ToUserName"),
                content="除了文本之外的消息还在开发中...😊"
            )
        serializer.save(**{
                "content":request.data.get("Content"),
                "msg_data_id":request.data.get("MsgDataId"),
                "idx":request.data.get("Idx"),
                "reply_content":reply
            })
        return Response(
            data=reply #
        )

    def get_renderers(self):
        if self.request.path=="/api/v1/wechat/public/message/" and self.request.method in ["GET","POST"] :
            return [CheckTokenRender()]
        return super().get_renderers()

    def retrieve(self, request, *args, **kwargs):
        """
            验证token
        """
        try:            
            data = request.query_params
            signature = data.get("signature")
            timestamp = data.get("timestamp")
            nonce = data.get("nonce")
            echostr = data.get("echostr")
            token = os.environ.get("WECHAT_URL_TOKEN")

            print("wechat check token",signature,nonce,echostr,token)
            list_ = [token, timestamp, nonce]
            list_.sort()
            sha1 = hashlib.sha1()
            map(sha1.update, list_)
            hashcode = sha1.hexdigest()
            print("hashcode",hashcode,"signature",signature)
            if hashcode == signature:
                return Response(
                    data=echostr
                )
            else:
                return Response(
                    data=echostr
                )
                # return HTTPResponse(
                #     status=status.HTTP_400_BAD_REQUEST,
                #     message=f"hashcode不等于signature"
                # )
        except Exception as exc:
            import traceback
            print("check wechat token error",str(exc),traceback.format_exc())
            return HTTPResponse(
                status=status.HTTP_400_BAD_REQUEST,
                message=str(exc)
            )