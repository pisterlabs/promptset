import openai
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

from PetMonitoringSystemBackend.settings import CHATGPT_CONFIG


class AdviceAPIView(APIView):
    @swagger_auto_schema(
        operation_summary='Advice Replier',
        operation_description='Give keyword as parameter to reply',
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'keyword': openapi.Schema(
                    type=openapi.TYPE_STRING
                )
            }
        )
    )
    def post(self, request, *args, **kwargs):
        keyword = request.data.get("keyword", "")
        if keyword and CHATGPT_CONFIG["enable"]:
            response = chat(keyword)
            return Response(data=response, status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_404_NOT_FOUND)

if CHATGPT_CONFIG["enable"]:
    def chat(data: str):
        openai.api_key = CHATGPT_CONFIG["api_key"]

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=data,
            max_tokens=512,
        )

        return response["choices"][0]["text"]


if __name__ == "__main__":
    print(chat("貓咪照顧方式建議"))
