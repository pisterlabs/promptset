from django.shortcuts import render, redirect
import openai

def initializeSession(request):
    request.session['messages'] = [
      {"role": "system",
        "content": "You are now chatting with a user, provide them with comprehensive, short and concise answers."},
  ]

def retrieveUserInput(request): 
      # get the prompt from the form
    prompt = request.POST.get('prompt')
    print("Q:" + str(prompt))
    # temperature is the randomness of the response under the AI context
    temperature = float(request.POST.get('temperature', 0.1))
    # append the prompt to the messages list
    request.session['messages'].append({"role": "user", "content": prompt})
    # set the session as modified
    request.session.modified = True
    # call the openai API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=request.session['messages'],
        temperature=temperature,
        max_tokens=1000,
    )
    
    return response, temperature

def formatResponse(request, response):
    # format the response
    formatted_response = response['choices'][0]['message']['content']
    # append the response to the messages list
    request.session['messages'].append({"role": "assistant", "content": formatted_response})
    request.session.modified = True

def createContext(request, temperature=1.0):
    # redirect to the home page
    context = {
        'messages': request.session['messages'],
        'prompt': '',
        'temperature': temperature,
    }
    
    return context 
