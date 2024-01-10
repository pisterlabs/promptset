import os
import openai
import json

from django.http import JsonResponse, HttpResponse, HttpRequest


openai.api_key = os.environ.get("OPENAI_APP_KEY ")



def openai_response(request: HttpRequest):

    if request.POST:
        data = json.loads(request.body)

        query = str(data['prompt'])

        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = query,
            temperature=0,
            max_tokens=130,
            frequency_penalty=0,
            presence_penalty=0

        )
        if "choices" in response:
            alternative_terms = {"sum":"addition", "multiplication":"product", "differnce":"subtraction"}
            output = response['choices'][0]['text']
            for ops in alternative_terms:
                operation_type = None
                new_query = set(query.split())
                if ops in new_query:
                    operation_type = ops.lower()
            return JsonResponse({
                "slackUsername": "Chiefkufre",
                "result": output,
                "operation_type": operation_type
            })
        