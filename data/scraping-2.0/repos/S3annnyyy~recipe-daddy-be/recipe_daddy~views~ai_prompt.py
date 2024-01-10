from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse
from rest_framework import status
import openai
import os

@api_view(['POST'])
def get_ai_prompt(request):
    openai.api_key = os.getenv("OPENAI_API_TOKEN")
    

    # Receive the message from the request body
    data = request.data
    user_message = data.get('userPrompt', '')
    schema = data.get('schema', '')
    
    try:
        if user_message and schema: print(f"Received userMsg and schema") 
            
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[                
                {"role": "system", "content": "You are a helpful recipe assistant. Only use the functions you have been provided with"},                
                {"role": "user", "content": user_message},
            ],
            functions=[{"name": "set_recipe", "parameters": schema, "outputs": schema}],
            function_call={"name": "set_recipe"}
        )

        print(response.choices[0])
       
        response_message = response["choices"][0]["message"]["function_call"]["arguments"]               

        # Return the generated text as JSON response.
        return JsonResponse({'generated_text': response_message}, status=200)        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)