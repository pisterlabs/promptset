# music_recommendations.py

import json
import openai

openai.api_key = 'API_KEY'

def extract_data_from_json(json_content):
    return json.loads(json_content)

def transform_data_for_recommendation(data):
    transformed_data = []
    for person in data:
        genre_preference = person['genre_preference']
        favorite_bands = person['favorite_bands']
        prompt = f"Recommend {genre_preference} music similar to {', '.join(favorite_bands)}."
        transformed_data.append((person, prompt))
    return transformed_data

def recommend_music(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def load_recommendations(recommendations):
    for person, recommendation in recommendations:
        print(f"Music recommendation for {person['name']}: {recommendation}")

def main():
    json_content = '''
    [
      {
        "id": 1,
        "name": "Bob",
        "age": 22,
        "genre_preference": "indie pop",
        "favorite_bands": ["Vampire Weekend", "Hippo Campus"],
        "location": "Los Angeles",
        "recently_attended_concert": "Tame Impala"
      },
      {
        "id": 2,
        "name": "Charlie",
        "age": 30,
        "genre_preference": "metal",
        "favorite_bands": ["Metallica", "Iron Maiden"],
        "location": "Chicago",
        "recently_attended_concert": "Slayer"
      },
      {
        "id": 3,
        "name": "Eva",
        "age": 28,
        "genre_preference": "pop",
        "favorite_bands": ["Taylor Swift", "Ariana Grande"],
        "location": "Miami",
        "recently_attended_concert": "Billie Eilish"
      }
    ]
    '''
    data = extract_data_from_json(json_content)
    transformed_data = transform_data_for_recommendation(data)
    recommendations = []
    for person, prompt in transformed_data:
        recommended_music = recommend_music(prompt)
        recommendations.append((person, recommended_music))
    load_recommendations(recommendations)

if __name__ == "__main__":
    main()
