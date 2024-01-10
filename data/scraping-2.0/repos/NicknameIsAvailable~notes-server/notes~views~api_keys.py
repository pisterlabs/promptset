from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import AccessToken
from openai import OpenAI
from notes.models import User
from rest_framework.response import Response
from rest_framework import status
from notes.models import OpenaiApiKey


class ApiKeys(APIView):
    def post(self, request, *args, **kwargs):
        token = AccessToken(request.headers["Authorization"].replace("Bearer ", ""))
        user_id = token.payload["user_id"]
        key = request.data["key"]
        client = OpenAI(
            api_key=key
        )

        try:
            generated_text = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Напиши короткий текст на любую тему до 10 слов"
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            user = User.objects.get(id=user_id)
            new_key = OpenaiApiKey(key=key, user=user)
            new_key.save()
            response = {"success": True, "message": "Ключ OpenAI сохранен", "key": key, "text": generated_text.choices[0].message.content}
            return Response(response, status=status.HTTP_200_OK, content_type="application/json")
        except Exception as e:
            print(e)
            response = {"success": True, "message": "Ключ не прошел валидацию!", "key": key, "error": str(e)}
            return Response(response, status=status.HTTP_400_BAD_REQUEST, content_type="application/json")

    def get(self, request, *args, **kwargs):
        token = AccessToken(request.headers["Authorization"].replace("Bearer ", ""))
        user_id = token.payload["user_id"]
        apikey = OpenaiApiKey.objects.get(user_id=user_id)

        response = {"success": True, "message": "Ключ получен", "data": apikey.to_json()}
        return Response(response, status=status.HTTP_200_OK, content_type="application/json")

