from .models import *
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.core.serializers import serialize
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from .serializers import *
import json
import requests
import openai


""" User Authentication """
def sign_up(request):
  email = request.data["email"]
  password = request.data["password"]
  super_user = False
  staff = False
  if 'super' in request.data:
      super_user = request.data['super']
  if 'staff' in request.data:
      staff = request.data['staff']

  # Validate email format
  try:
      validate_email(email)
  except ValidationError as ve:
      print(ve)
      return JsonResponse({"success": False, "error": "Invalid email format"})

  try:
      # create_user() auto-hashes password
      new_user = App_User.objects.create_user(username=email, email=email, password=password, is_superuser=super_user, is_staff=staff)
      new_user.save()
      return JsonResponse({"success": f"{email} was created."})
  except Exception as e:
      print(e)
      return JsonResponse({"success": False})

def sign_in(request):
  email = request.data["email"]
  password = request.data["password"]
  user = authenticate(username = email, password = password)
  # once user is authenticated, they need to login
  if user is not None and user.is_active:
    try:
      # request._request passes in a WSGI request, not a QuerySet request like normal django rest framework
      # generates csrftoken and sessionid cookies
      login(request._request, user)
      return JsonResponse({'email': user.email})
    except Exception as e:
      print(e)
      return JsonResponse({"sign_in": False})
  # makes sure something is returned if conditional fails  
  return JsonResponse({"sign_in": False})

def current_user(request):
  if request.user.is_authenticated:
    # serialize takes db query data (i.e. QuerySet) and makes it readable via json
    # json.loads accesses the json object
    # **options (see def of serialize()) are the keys from the object passed in. "fields" variable is necessary syntax/variable name
    user_info = serialize("json", [request.user], fields = ["email"])
    user_info_workable = json.loads(user_info)    
    # user_info[0] digs into first elem of list, but slightly unnecessary because list will always have 1 elem. QoL though, one level is already dug into
    return JsonResponse(user_info_workable[0]["fields"])
  else: 
    return JsonResponse({"user": None})
  
# def sign_out(request):
#   try:
#     # purges authorization header (i.e. sessionid cookie) from request
#     logout(request)
#     return JsonResponse({"sign_out": True})
#   except Exception as e:
#     print(e)
#     return JsonResponse({"sign_out": False})
  
""" Quote + Recommendation """
# helper function to define mechanism of OpenAI API call
def get_recommendations(messages_with_quote):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages_with_quote
  )
  
  # parse the response to separate user-facing and system-facing data
  response_content = completion['choices'][0]['message']['content']
  
  # If only one value is returned (i.e. user input was invalid), return the response_content directly
  try:
    conversational_response, system_response = response_content.split("end_response")
  except ValueError:
    return response_content, None

  # clean whitespace of user-facing data, passed to front-end
  conversational_response = conversational_response.strip()
  # system-facing data, inserted into db
  system_response = system_response.strip()

  return conversational_response, system_response

# helper function to define mechanism of Google Books API call
def get_book(title, author, google_books_api_key):
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}+inauthor:{author}&key={google_books_api_key}"
    response = requests.get(url)
    data = response.json()

    if data['totalItems'] > 0:
      # set to first (i.e. most relevant) search result
        book = data['items'][0]
        return book
    else:
        # refactor for better error handling
        return None
  
# helper function to save records in db
def save_records(user, quote_text, system_response):
  quote = Quote.objects.create(user_id = user, quote_text = quote_text)

  # loop through the list given by second API response, and create Recommendation records
  for recommendation in system_response:
      Recommendation.objects.create(
          quote_id=quote,
          title=recommendation['title'],
          author=recommendation['author'],
          summary=recommendation['summary'],
          date_published=recommendation['date_published'],
          google_books_link=recommendation.get('google_books_link', '')
      )
      
def user_recommendation_history(request):
  try: 
    # retrieve correct user from db to establish links
    user_email = request.GET.get("user_email")
    user = App_User.objects.get(email=user_email)
    # sort by created_at to display most recent quotes to the user at the front-end
    quotes = Quote.objects.filter(user_id=user).order_by('-created_at')
    quote_data = []

    # for all quotes submitted by that user, grab their recommendations 
    for quote in quotes:
      recommendations = Recommendation.objects.filter(quote_id=quote)
      
      # Use the custom serializers to grab pk
      quote_serializer = CustomQuoteSerializer(quote)
      serialized_quote = json.dumps(quote_serializer.data)

      recommendation_serializer = CustomRecommendationSerializer(recommendations, many=True)
      serialized_recommendations = json.dumps(recommendation_serializer.data)
      
      # manipulate the objects to send only the necessities to the front-end
      quote_json = json.loads(serialized_quote)
      recommendations_json = json.loads(serialized_recommendations)

      # attach combined data points via a dict and send forward
      quote_data.append({
          "quote": quote_json,
          "recommendations": recommendations_json
      })
          
    # print(quote_data)
    return JsonResponse({ "history": quote_data })
  except Exception as e:
    print("User History error:", e)
    return JsonResponse({ "success": False })
  
def delete_recommendation(request):
  try:
    recommendation_id = request.data["recommendation_pk"]
    recommendation = Recommendation.objects.get(id=recommendation_id)
    recommendation.delete()
    
    return JsonResponse({"success": True})
  except Exception as e:
    print(e)
    return JsonResponse({"success": False})

def delete_quote(request):
  try:
    quote_id = request.data["quote_pk"]
    quote = Quote.objects.get(id=quote_id)
    quote.delete()
    
    return JsonResponse({"success": True})
  except Exception as e:
    print(e)
    return JsonResponse({ "success": False })