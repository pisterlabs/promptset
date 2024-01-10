import io
import time
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from decouple import config
import openai
from drf_yasg import openapi

class GptApiView(APIView):
    @swagger_auto_schema(
        operation_description="Enviar uma mensagem para API do ChatGPT e retorna a resposta.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='Pergunta feita pelo usuário.'),
                'behavior': openapi.Schema(type=openapi.TYPE_INTEGER, 
                                           enum=[1,2,3],
                                           description="Comportamento do assistente de IA. Sendo 1 para dicas, 2 para explicação de erro e 3 para explicação de código.")
            },
            required=['message', 'behavior']
        ),
        responses={
            200: openapi.Response(
                description='Sucesso',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'response': openapi.Schema(type=openapi.TYPE_STRING, description='Resposta do modelo.'),
                    }
                ),
            ),
        }
    )
    def post(self, request):
        openai.api_key = config('OPENAI')
        userMsg = self.request.data['message']
        gptBehavior = self.request.data['behavior']

        behaviors = {
            1 : 'Você irá somente dar dicas simples, não forneca código corrigido',
            2 : 'Dada a descrição do problema, a entrada e saída e o código com erro, explique passo a passo como corrigir o código levando também em consideração, porém, sem fornecer o código corrigido.',
            3 : 'Explique linha a linha do código de forma resumida'
        }

        chat = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{"role": "system", "content": behaviors[gptBehavior]},
                        {"role": "user", "content": userMsg}
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response = chat["choices"][0]["message"]["content"]
        return Response(data={'response':response})