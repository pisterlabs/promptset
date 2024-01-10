import boto3
import os
from openai import OpenAI
import uuid
from data.models import Meal, Restaurant
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)

class Classification(Enum):
   RECIPE = 1
   RESTAURANT = 2
   EITHER = 3
   
class RecommendationService:
  # todo log ingredients
  # for cooking inject ingredients

  # todo for restaurants inject new that other users enjoyed based on user's preferences
  # todo inject based on location
  # todo inject based on what's retrieved from embeddings

  # todo link to reservations or togo ordering links
  # todo link to instacart 
  # todo create shopping list and add there for user?

  # todo move model names to config values

  def __init__(self, db):
    self.db = db
    self.s3_client = boto3.client('s3', aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.environ.get("AWS_SECRET_KEY_ID"), region_name="us-west-2")
    # OpenAI API configuration
    self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY_NAME"))

  def get_restaurants(self, user_id, page_number=1, page_size=10):
     # todo index
     # todo sort by location from user's request?
     offset = (page_number - 1) * page_size
     restaurants = Restaurant.query.filter_by(deleted=False).limit(page_size).offset(offset).all()
     restaurants_obj = [restaurant.to_dict() for restaurant in restaurants]
     total_restaurants_count = Restaurant.query.filter_by(deleted=False).count()
     total_pages = (total_restaurants_count + page_size - 1) // page_size

     return {
        "restaurants": restaurants_obj,
        "total_pages": total_pages,
        "current_page": page_number,
        "page_size": page_size
     }

  def add_restaurant(self, request):
    price = request.form.get('price')
    restaurant_name = request.form.get('restaurant_name')
    cuisine = request.form.get('cuisine')
    description = request.form.get('description')
    embedding = self._get_openai_embedding(description)
    if embedding is None:
        raise Exception("Unable to generate embeddings")

    restaurant = Restaurant(
        price=price, 
        cuisine=cuisine, 
        description=description,
        name=restaurant_name
    )
    self.db.session.add(restaurant)
    self.db.session.commit()
    return {"status": "success"}
    raise NotImplementedError("add_restaurant must be implemented")


  def delete_meal(self, request, user_id):
     id = request.get("id")
     meal = Meal.query.filter_by(id=id, user_id=user_id).first()
     self.db.sessions.delete(meal)
     self.db.session.commit()
     return {"status": "success"}
  
  def get_meals(self, user_id, page_number=1, page_size=10):
     # todo index
     # todo get all meals not just user's?
     offset = (page_number - 1) * page_size
     meals = Meal.query.filter_by(user_id=user_id, deleted=False).limit(page_size).offset(offset).all()
     meals_obj = [meal.to_dict() for meal in meals]
     total_meals_count = Meal.query.filter_by(user_id=user_id).count()
     total_pages = (total_meals_count + page_size - 1) // page_size

     return {
        "meals": meals_obj,
        "total_pages": total_pages,
        "current_page": page_number,
        "page_size": page_size
     }
      
  def add_meal(self, request, user_id):
     # todo return  meal
    price = request.form.get('price')
    restaurant_id = request.form.get('restaurant_id')
    cuisine = request.form.get('cuisine')
    description = request.form.get('description')
    meal_name = request.form.get('mealName')
    review = request.form.get('review')    
    image_urls = []
    if 'images' in request.files:
        for image in request.files.getlist('images'):
            image_url = self._upload_image_to_s3(image)
            if image_url:
                image_urls.append(image_url)
    # Generate embeddings for the description
    embedding = self._get_openai_embedding(description)
    if embedding is None:
        raise Exception("Unable to generate embeddings")

    # todo restaurant_id 
    # embedding
    meal = Meal(
        user_id=user_id,
        restaurant_id=restaurant_id,
        price=price, 
        cuisine=cuisine, 
        description=description, 
        image_urls=image_urls,
        homecooked=False,
        name=meal_name
    )
    self.db.session.add(meal)
    self.db.session.commit()
    return {"status": "success"}
  
  def _classify(self, msg):
     completion = self.openai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "Classify the user's message to recommend either a recipe, a restaurant or either. Return just recipe if the user is looking for a recipe. Return just restaurant if the user is looking for a restaurant recommendation."},
        {"role": "user", "content": msg}
      ]
      )
     resp = completion.choices[0].message
     if resp.content == "recipe":
        return Classification.RECIPE
     elif resp.content == "restaurant":
        return Classification.RESTAURANT
     return Classification.EITHER

  def _assemble_restaurant_recommendation_context(self, lat, long, location):
     # todo fetch nearby restaurants
     # fetch based on preference
     return f"You are Sous a helpful dining assistant. Your goal is to help a user pick what to eat being as specific as possible. Give multiple restaurant recommendations in {location} and describe what to order at each. Limit the response to around 60 tokens and only recommend restaurants. Use bullet points."

  def _assemble_recipe_recommendation_context(self):
     # todo fetch recipes based on embeddings
     # todo fetch ingredients user has
     # todo fetch dietary preferences and factor in
     return "You are Sous a helpful cooking assistant. Your goal is to help a user pick what to eat being as specific as possible. Give recommendations based on the user's inquiry. For what to cook format the recipe and ingredients so it's easy to purchase and follow. Limit the response to around 150 tokens and only recommend recipes."
  
  def _assemble_default_context(self):
     return "You are Sous a helpful dining and cooking assistant. Your goal is to help a user pick what to eat being as specific as possible. Give recommendations based on the user's inquiry. For what to cook format the recipe and ingredients so it's easy to purchase and follow. For restaurants give multiple options and describe what to order. Limit the response to around 30 tokens and give both dining out and recipe recommendations."

  def _assemble_context(self, classification, lat, long, location):
     if classification == Classification.RECIPE:
        return self._assemble_recipe_recommendation_context()
     elif classification == Classification.RESTAURANT:
        ctx = self._assemble_restaurant_recommendation_context(lat, long, location)
        return ctx
     return self._assemble_default_context()

  def get_recommendation(self, request, user_id):
    # todo add moderation
    print(request)
    # todo get default from user profile
    lat = request.get("latitude")
    lon = request.get("longitude")
    location = request.get("location_name")
    classification = self._classify(request.get('message'))
    print(f"classification: {classification}")
    completion = self.openai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": self._assemble_context(classification, lat, lon, location)},
        {"role": "user", "content": request.get('message')}
      ]
    )

    resp = completion.choices[0].message
    return {"message": resp.content}
    raise NotImplementedError("get_recommendation must be implemented")
     
  def _upload_image_to_s3(self, file):
    try:
        file_key = str(uuid.uuid4())  # Generate unique file name
       
        self.s3_client.upload_fileobj(file, os.environ.get("IMAGE_BUCKET_NAME"), file_key)
        return f'https://{os.environ.get("IMAGE_BUCKET_NAME")}.s3.amazonaws.com/{file_key}'
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None
       
  def _get_openai_embedding(self, text):
    try:
        response = self.openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
